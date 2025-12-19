"""
Tianji项目主模块初始化文件

该模块负责初始化Tianji项目的主路径和配置，确保项目正确安装和运行。
"""
from pathlib import Path
from loguru import logger
import tianji


def get_tianji_package_path():
    """
    获取已安装包的根目录
    
    通过检查.git和.gitignore文件来确定项目根目录，
    如果找不到则使用当前工作目录。
    
    Returns:
        Path: Tianji项目的根目录路径
        
    Raises:
        FileNotFoundError: 如果当前目录缺少必要的项目文件
    """
    # 获取tianji模块的父目录作为包根目录
    package_root = Path(tianji.__file__).parent.parent
    
    # 检查是否存在git相关文件来确定项目根目录
    for i in (".git", ".gitignore"):
        if (package_root / i).exists():
            break
    else:
        # 如果找不到git文件，使用当前工作目录
        package_root = Path.cwd()

    # 验证项目结构完整性
    if (
        not (package_root / ".gitignore").exists()
        or not (package_root / "tianji").exists()
        or not (package_root / "run").exists()
    ):
        raise FileNotFoundError(
            "当前目录缺少 .gitignore 文件或 tianji 目录，可能不是根目录，请重新 `pip install -e .` 安装!"
        )
    
    # 记录初始化信息
    logger.info(f"初始化完毕,当前执行根目录为 {str(package_root)}")
    return package_root


# 全局变量：存储Tianji项目根路径
TIANJI_PATH = get_tianji_package_path()
