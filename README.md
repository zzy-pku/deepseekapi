# deepseekapi
高考志愿填报助手
集成了deepseekapi的高考志愿填报助手

文件解释：
ds.py：核心程序，源代码
import_csv_to_knowledge.py:用于将csv数据转换为本地数据库，与运行代码无关
requirements.txt：环境依赖

部署说明:1、下载ds.py，数据库
        2、新建环境并将requirements.txt中要求的依赖库导入
        3、创建.env文件在ds同目录下，并在文件中编写DEEPSEEK_API_KEY=你的deepseekapi密钥,以及KNOWLEDGE_DB_PATH=你的数据库文件路径
        4、最后科学上网即可运行代码

数据库下载链接：https://disk.pku.edu.cn/link/AAB177D0A10B15434CBF204B94933FC61A
文件名：knowledge_base.db
有效期限：2025-07-27 01:06
