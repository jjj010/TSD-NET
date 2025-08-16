# utils/profiler.py
import time
import torch
import torch.nn as nn
from collections import defaultdict
import pandas as pd
import os
from contextlib import contextmanager

class StageBucket:
    def __init__(self):
        # ── 结构级（stage）统计（保持你的原有逻辑） ──
        self.params = 0
        self.macs = 0
        self.last_time = 0.0           # 本次 forward 的耗时（stage 级）
        self.peak_mem_delta = 0        # 本次 forward 的显存峰值增量（stage 级，仅 forward）
        # 累积统计（用于 epoch 平均）
        self.sum_time_train = 0.0
        self.sum_time_eval  = 0.0
        self.calls_train = 0
        self.calls_eval  = 0

class PhaseBucket:
    """训练三段统计（forward / backward / optim）"""
    def __init__(self):
        self.last_time = 0.0
        self.sum_time = 0.0
        self.calls = 0
        self.peak_mem_delta = 0  # 本段内 CUDA 峰值增量

class StageProfiler:
    def __init__(self, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # ── stage 级统计（前向内部细分） ──
        self.stats = defaultdict(StageBucket)
        self._t_start = {}
        self._mem_start = {}
        self._mac_bucket_current = 0
        self._stage_stack = []           # 嵌套 stage 支持
        self._hooks = []

        # ── 训练三段统计（batch 粒度） ──
        self._phase_names = ("forward", "backward", "optim")
        self._phases = {n: PhaseBucket() for n in self._phase_names}
        self._phase_t0 = {n: 0.0 for n in self._phase_names}
        self._phase_mem0 = {n: 0 for n in self._phase_names}

        # 整个 batch 的峰值显存
        self.last_batch_peak_mem = 0
        self.max_batch_peak_mem = 0

    # --------- 实用函数 ---------
    def _cuda_available(self):
        return self.device.type == 'cuda' and torch.cuda.is_available()

    def _mem_now(self):
        if self._cuda_available():
            torch.cuda.synchronize(self.device)
            return torch.cuda.max_memory_allocated(self.device)
        return 0

    def reset_cuda_peak(self):
        if self._cuda_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    # ============================================================
    # =============== 一、结构级（stage）统计 =====================
    # ============================================================
    def register_stage(self, module: nn.Module, stage_name: str):
        """为某个模块注册一个“阶段”，统计其前向耗时、峰值显存增量、以及把基础算子 MACs 归到该阶段。"""
        def _pre(_, _inp):
            if self._cuda_available():
                torch.cuda.synchronize(self.device)
            self._t_start[stage_name] = time.perf_counter()
            # 为了获得“阶段内峰值”，进入阶段时先 reset，再在退出时读 max
            if self._cuda_available():
                torch.cuda.reset_peak_memory_stats(self.device)
            self._mem_start[stage_name] = self._mem_now()

            self._stage_stack.append(stage_name)
            self._mac_bucket_current = 0

        def _post(_, _inp, _out):
            if self._cuda_available():
                torch.cuda.synchronize(self.device)
            dt = time.perf_counter() - self._t_start.get(stage_name, time.perf_counter())

            # 峰值增量（阶段内）
            curr_peak = self._mem_now()
            mem_base = self._mem_start.get(stage_name, 0)
            mem_delta = max(0, curr_peak - mem_base)

            sb = self.stats[stage_name]
            sb.last_time = dt
            sb.peak_mem_delta = max(sb.peak_mem_delta, mem_delta)

            # 归集 MACs
            sb.macs += self._mac_bucket_current
            self._mac_bucket_current = 0
            if self._stage_stack and self._stage_stack[-1] == stage_name:
                self._stage_stack.pop()

            # train / eval 累积
            mode = "train" if module.training else "eval"
            if mode == "train":
                sb.sum_time_train += dt
                sb.calls_train += 1
            else:
                sb.sum_time_eval += dt
                sb.calls_eval += 1

        self._hooks.append(module.register_forward_pre_hook(_pre))
        self._hooks.append(module.register_forward_hook(_post))

    def add_params_of(self, module: nn.Module, stage_name: str):
        p = sum(p.numel() for p in module.parameters() if p.requires_grad)
        self.stats[stage_name].params += p

    def register_macs_hooks(self, model: nn.Module):
        """注册基础算子 MACs 近似（仅前向）"""
        def _mac_linear(m: nn.Linear, inp, out):
            if not self._stage_stack:
                return
            out_nelem = out.numel()
            out_f = m.out_features
            in_f = m.in_features
            macs = (out_nelem // out_f) * in_f * out_f
            self._mac_bucket_current += macs

        def _mac_conv1d(m: nn.Conv1d, inp, out):
            if not self._stage_stack:
                return
            out_elems = out.numel()
            kernel_mul = (m.in_channels // m.groups) * m.kernel_size[0]
            macs = out_elems * kernel_mul
            self._mac_bucket_current += macs

        def _mac_gru(m: nn.GRU, inp, out):
            if not self._stage_stack:
                return
            x = inp[0]
            batch_first = getattr(m, 'batch_first', False)
            if batch_first:
                B, L, I = x.shape
            else:
                L, B, I = x.shape
            H = m.hidden_size
            layers = m.num_layers
            dirs = 2 if m.bidirectional else 1
            macs = L * B * layers * dirs * (3 * (I * H + H * H))
            self._mac_bucket_current += macs

        for m in model.modules():
            if isinstance(m, nn.Linear):
                self._hooks.append(m.register_forward_hook(_mac_linear))
            elif isinstance(m, nn.Conv1d):
                self._hooks.append(m.register_forward_hook(_mac_conv1d))
            elif isinstance(m, nn.GRU):
                self._hooks.append(m.register_forward_hook(_mac_gru))

    # ============================================================
    # =============== 二、训练三段（batch 粒度） ==================
    # ============================================================
    def _phase_begin(self, name: str):
        if self._cuda_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._phase_t0[name] = time.perf_counter()
        self._phase_mem0[name] = self._mem_now()

    def _phase_end(self, name: str):
        if self._cuda_available():
            torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - self._phase_t0[name]
        peak = self._mem_now()
        base = self._phase_mem0[name]
        delta = max(0, peak - base)

        pb = self._phases[name]
        pb.last_time = dt
        pb.sum_time += dt
        pb.calls += 1
        pb.peak_mem_delta = max(pb.peak_mem_delta, delta)

    @contextmanager
    def fwd(self):
        """with profiler.fwd():  output = model(x)"""
        self._phase_begin("forward")
        try:
            yield
        finally:
            self._phase_end("forward")

    @contextmanager
    def bwd(self):
        """with profiler.bwd():  loss.backward()"""
        self._phase_begin("backward")
        try:
            yield
        finally:
            self._phase_end("backward")

    @contextmanager
    def opt(self):
        """with profiler.opt():  optimizer.step()"""
        self._phase_begin("optim")
        try:
            yield
        finally:
            self._phase_end("optim")

    def batch_begin(self):
        """在一个 batch 开始时调用，用于记录该 batch 的整体峰值显存。"""
        if self._cuda_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def batch_end(self):
        """在一个 batch 结束时调用，记录该 batch 整体的峰值显存。"""
        if self._cuda_available():
            torch.cuda.synchronize(self.device)
            peak = self._mem_now()
            self.last_batch_peak_mem = peak
            self.max_batch_peak_mem = max(self.max_batch_peak_mem, peak)

    # ============================================================
    # ========================= 汇总输出 ==========================
    # ============================================================
    def summarize(self, save_csv_path=None, include_epoch_avg=True):
        """
        返回 pandas.DataFrame，包含（stage 级）：
          Stage, Params, MACs(approx), PeakMemΔ, Time(s)  [Time(s) = 最近一次 forward]
        如果 include_epoch_avg=True，同时包含：
          Calls(train), AvgTime(train), Calls(eval), AvgTime(eval)

        （注意：这是“结构级前向”视角，不含 backward/optim）
        """
        rows = []
        for name, sb in self.stats.items():
            row = {
                "Stage": name,
                "Params": sb.params,
                "MACs(approx)": sb.macs,
                "PeakMemΔ": sb.peak_mem_delta,
                "Time(s)": sb.last_time,
            }
            if include_epoch_avg:
                avg_tr = (sb.sum_time_train / sb.calls_train) if sb.calls_train else 0.0
                avg_ev = (sb.sum_time_eval / sb.calls_eval) if sb.calls_eval else 0.0
                row.update({
                    "Calls(train)": sb.calls_train,
                    "AvgTime(train)": avg_tr,
                    "Calls(eval)": sb.calls_eval,
                    "AvgTime(eval)": avg_ev,
                })
            rows.append(row)

        df = pd.DataFrame(rows, columns=[
            "Stage","Params","MACs(approx)","PeakMemΔ","Time(s)",
            "Calls(train)","AvgTime(train)","Calls(eval)","AvgTime(eval)"
        ])
        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            df.to_csv(save_csv_path, index=False)
        return df

    def summarize_training_phases(self, save_csv_path=None):
        """
        返回 pandas.DataFrame，包含（batch 粒度，三段统计）：
          Phase, Calls, LastTime(s), AvgTime(s), PeakMemΔ(max), LastBatchPeakMem, MaxBatchPeakMem
        注：PeakMemΔ(max) 为该段内历史最大峰值增量；LastBatchPeakMem/MaxBatchPeakMem 为整 batch 视角。
        """
        rows = []
        for name in self._phase_names:
            pb = self._phases[name]
            avg = (pb.sum_time / pb.calls) if pb.calls else 0.0
            rows.append({
                "Phase": name,
                "Calls": pb.calls,
                "LastTime(s)": pb.last_time,
                "AvgTime(s)": avg,
                "PeakMemΔ(max)": pb.peak_mem_delta,
                "LastBatchPeakMem": self.last_batch_peak_mem,
                "MaxBatchPeakMem": self.max_batch_peak_mem,
            })

        df = pd.DataFrame(rows, columns=[
            "Phase","Calls","LastTime(s)","AvgTime(s)","PeakMemΔ(max)",
            "LastBatchPeakMem","MaxBatchPeakMem"
        ])
        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            df.to_csv(save_csv_path, index=False)
        return df

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
