# 针对 reduce 的系列优化

参考资料：
[developer.download.nvidia.com](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

Gemini


内容：

[ReduceV0](https://famfc6p4iwg.feishu.cn/docx/VIJcd2qGiojcQUxkU1Lc5XeVnHb)，写出一个可以正确运行 reduce 代码

[ReduceV1](https://famfc6p4iwg.feishu.cn/docx/L1ModdfZjoiwIKxCK8ZcilSUnCb)，发现 V0 版本的分支和取模计算过于浪费资源，通过 index 映射 tid 来规约

[ReduceV2](https://famfc6p4iwg.feishu.cn/docx/GMWndfCB3o8X86xr5ogcfec0nje)，V1 规约的过程出现了 Bank Conflict 冲突，修改映射方式（stride 由大变小）来避开 Bank Conflict

[ReduceV3](https://famfc6p4iwg.feishu.cn/docx/DKIydl70soIpqBxrvqNc2k6anGf)，在读取过程中提前进行一次加法，并且利用 GPU 的 ILP 来掩盖延迟

[ReduceV4](https://famfc6p4iwg.feishu.cn/docx/KQi2dtxpiouhd1xFfq0chV2bnac)，利用 Warp Shuffle 把最后几次循环展开，减少 kernel 等待和循环计算

[ReduceV5](https://famfc6p4iwg.feishu.cn/docx/Vs9Zd51wNoLaGfx97aNcGcxUnVh)，穷举 Block 支持的线程数，再次消除 kernel 内部的循环（模板编程）

[ReduceV6](https://famfc6p4iwg.feishu.cn/docx/UrszdEEZBojgXhxOp6Pccw5snmh)，根据 Brent's Theorem 让线程串行累加来掩盖 DRAM 延时

[ReduceV7](https://famfc6p4iwg.feishu.cn/docx/Uq33dCeeqo7Z6KxF53icovIEnbg)，根据访存向量化和 ILP 来优化 DRAM 瓶颈

[CUB Reduce](https://famfc6p4iwg.feishu.cn/docx/I1rldNQOIomOGexAkDic2p6znre)，NVIDIA CUB 官方的一些优化思路，Atomic Counter，Decoupled Look-back

README 可能更新不及时，查看飞书文档 [Reduce](https://famfc6p4iwg.feishu.cn/docx/KNyJdFz4Ho9WFLxxxIRcplOfnId)

# 针对 GEMM 的系列优化