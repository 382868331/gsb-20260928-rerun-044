"""统计试验运行平台（Statistical Experiment Runner Platform）。

统一管理：
- 带版本的人工数据；
- 检验设定、随机种子与原假设；
- 批次（batch）下的多项检验及其完整运行结果；
- 可中断、可恢复、不重复计算的运行过程。

第一项能力：带分组（区组/层）结构数据的置换检验。
后续新增检验类型只需在 ``statplatform.registry`` 注册执行器，
仍走同一套批次与结果记录。
"""

from .batch import Platform
from .data import DataBundle
from .permutation import PermutationSpec

__all__ = ["Platform", "DataBundle", "PermutationSpec"]
__version__ = "0.1.0"
