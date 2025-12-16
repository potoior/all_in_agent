"""FastAPI 文件上传处理示例
演示了如何处理单文件和多文件上传，包括bytes和UploadFile两种方式
"""

from fastapi import FastAPI, File, UploadFile

# 创建 FastAPI 应用实例
app = FastAPI()


# 单文件上传示例
@app.post("/files/")
async def create_file(file: bytes = File(description="A file read as bytes")):
    """使用bytes方式处理单文件上传
    
    这种方式会将整个文件读入内存，适用于小文件
    
    Args:
        file (bytes): 上传的文件数据
        
    Returns:
        dict: 包含文件大小的字典
    """
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(
    file: UploadFile = File(description="A file read as UploadFile"),
):
    """使用UploadFile方式处理单文件上传
    
    UploadFile提供了更多文件信息和流式处理能力，适用于大文件
    
    Args:
        file (UploadFile): 上传的文件对象
        
    Returns:
        dict: 包含文件名的字典
    """
    return {"filename": file.filename}


# 多文件上传示例
# 重新导入必要的模块（因为在前面已经导入过，此处为示例完整性）
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

# 重新创建应用实例（实际项目中不应重复创建）
app = FastAPI()


@app.post("/files/")
async def create_files(
    files: list[bytes] = File(description="Multiple files as bytes"),
):
    """使用bytes方式处理多文件上传
    
    Args:
        files (list[bytes]): 上传的文件数据列表
        
    Returns:
        dict: 包含各文件大小的字典
    """
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(
    files: list[UploadFile] = File(description="Multiple files as UploadFile"),
):
    """使用UploadFile方式处理多文件上传
    
    Args:
        files (list[UploadFile]): 上传的文件对象列表
        
    Returns:
        dict: 包含各文件名的字典
    """
    return {"filenames": [file.filename for file in files]}


@app.get("/")
async def main():
    """提供文件上传的HTML表单页面
    
    Returns:
        HTMLResponse: 包含文件上传表单的HTML页面
    """
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)