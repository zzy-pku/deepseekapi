import os
from dotenv import load_dotenv
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import requests
import json
from jsonschema import validate, ValidationError
import gradio as gr
import time
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam
)
import numpy as np
from PIL import Image
from io import BytesIO
import faiss
from sentence_transformers import SentenceTransformer
import sqlite3
from typing import List, Dict, Any, Optional, Union
import base64
from simpleeval import simple_eval
from transformers import BlipProcessor, BlipForConditionalGeneration
import whisper

# 配置DeepSeek API密钥（请在.env文件中设置DEEPSEEK_API_KEY）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com"

# 多模态模型配置
MULTIMODAL_MODELS = {
    "deepseek-chat-vision": "支持图像输入的对话模型",
    "deepseek-audio-chat": "支持语音输入输出的模型"
}

# 初始化OpenAI客户端（指向DeepSeek API）
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_URL
)

# 定义输入输出规范（JSON Schema）
input_schema = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "model": {"type": "string", "enum": ["deepseek-chat", "deepseek-chat-large"]},  # 替换为DeepSeek模型
        "temperature": {"type": "number", "minimum": 0, "maximum": 2},
        "max_tokens": {"type": "integer", "minimum": 1, "maximum": 8000}  # DeepSeek可能支持更大token数
    },
    "required": ["prompt"]
}

output_schema = {
    "type": "object",
    "properties": {
        "response": {"type": "string"},
        "model": {"type": "string"},
        "temperature": {"type": "number"},
        "tokens_used": {"type": "integer"}
    },
    "required": ["response"]
}

# 1. 定义高考志愿输出的 JSON Schema（score_comparison 只要求 last_year_min）
GAOKAO_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_analysis": {"type": "string"},
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "university": {"type": "string"},
                    "major": {"type": "string"},
                    "probability": {"type": "string", "enum": ["冲刺", "稳妥", "保底"]},
                    "score_comparison": {
                        "type": "object",
                        "properties": {
                            "last_year_min": {"type": "number"}
                        },
                        "required": ["last_year_min"]
                    }
                },
                "required": ["university", "major", "probability", "score_comparison"]
            }
        },
        "strategy_advice": {"type": "string"}
    },
    "required": ["candidate_analysis", "recommendations", "strategy_advice"]
}

# 初始化RAG组件
try:
    EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # 多语言公开模型
except Exception as e:
    print(f"向量模型加载失败: {e}")
    EMBEDDING_MODEL = None
faiss_index = faiss.IndexFlatL2(EMBEDDING_MODEL.get_sentence_embedding_dimension())
knowledge_db = sqlite3.connect("knowledge_base.db")
knowledge_cur = knowledge_db.cursor()
knowledge_cur.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    content TEXT,
    metadata TEXT,
    embedding BLOB
)
''')
knowledge_db.commit()

SENSITIVE_WORDS = ["傻逼", "fuck", "操你", "敏感词1", "敏感词2"]  # 可根据实际需求扩展
INJECTION_PATTERNS = ["<script", "</script>", "{{", "}}", "{%", "%}", "os.system", "subprocess", "import os", "import sys"]

# 全局只加载一次BLIP2模型
BLIP2_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
BLIP2_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 全局只加载一次 Whisper 模型
WHISPER_MODEL = whisper.load_model("base")

def preprocess_input_text(text: str) -> str:
    """对输入文本进行敏感词过滤和指令注入防护，发现问题则替换或阻断。"""
    # 敏感词过滤
    for word in SENSITIVE_WORDS:
        if word in text:
            text = text.replace(word, "*")
    # 指令注入防护（简单检测，发现危险片段则阻断）
    for pattern in INJECTION_PATTERNS:
        if pattern.lower() in text.lower():
            return "检测到潜在的指令注入风险，输入已被阻断。"
    return text

# ---------------------
# 多模态数据处理
# ---------------------
def process_image(image: Image.Image) -> Dict[str, Any]:
    """处理图像输入并转换为API所需格式"""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # DeepSeek多模态API要求的图像格式
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
        }
    }


def process_audio(audio_file: str) -> Dict[str, Any]:
    """将本地音频文件转为 base64 并按 DeepSeek API 格式返回"""
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return {
        "type": "audio_url",
        "audio_url": {
            "url": f"data:audio/wav;base64,{audio_base64}",
            "content_type": "audio/wav"
        }
    }


def generate_audio_response(text: str, model: str) -> bytes:
    """生成语音输出"""
    # 调用DeepSeek语音合成API
    response = client.audio.speech.create(
        model=model,
        input=text,
        voice="default"
    )
    return response.content


# ---------------------
# 结构化输出控制
# ---------------------
def validate_structure(output: str, schema: dict = GAOKAO_OUTPUT_SCHEMA) -> bool:
    try:
        data = json.loads(output)
        validate(instance=data, schema=schema)
        return True
    except Exception as e:
        print(f"结构化校验异常: {str(e)}")
        return False

def enforce_structure(prompt: str, schema: dict = GAOKAO_OUTPUT_SCHEMA, max_retries: int = 3, user_id: str = "default_user") -> str:
    structured_prompt = f"""
    你是一个高考志愿填报专家，请严格按照以下JSON格式生成响应:
    {json.dumps(schema, ensure_ascii=False, indent=2)}
    
    考生信息: {prompt}
    
    要求:
    1. 必须包含candidate_analysis和recommendations
    2. 每个推荐必须包含university、major和probability
    3. probability只能是\"冲刺\"、\"稳妥\"或\"保底\"
    4. 每个推荐必须包含score_comparison字段，内容包括last_year_min（去年最低分）
    5. 必须包含strategy_advice字段，内容为对考生的填报策略建议
    """
    for attempt in range(max_retries):
        response = generate_response({
            "prompt": structured_prompt,
            "model": "deepseek-chat",
            "response_format": {"type": "json_object"}
        }, user_id)
        if validate_structure(response["response"], schema):
            return response["response"]
        structured_prompt = f"""
        上次响应格式有误，请重新按照以下JSON格式生成:
        {json.dumps(schema, ensure_ascii=False, indent=2)}
        
        之前的响应: {response["response"]}
        
        考生信息: {prompt}
        4. 每个推荐必须包含score_comparison字段，内容包括last_year_min（去年最低分）
        5. 必须包含strategy_advice字段，内容为对考生的填报策略建议
        """
    return json.dumps({"error": "无法生成符合格式的响应", "raw_output": response["response"]})


# ---------------------
# RAG架构实现
# ---------------------
def embed_text(text: str) -> np.ndarray:
    """将文本转换为向量表示"""
    embedding = EMBEDDING_MODEL.encode(text)
    return embedding.astype(np.float32)


def store_knowledge(content: str, metadata: Dict[str, Any] = None) -> int:
    """存储知识到知识库"""
    embedding = embed_text(content)
    knowledge_cur.execute(
        "INSERT INTO documents (content, metadata, embedding) VALUES (?, ?, ?)",
        (content, json.dumps(metadata or {}), embedding.tobytes())
    )
    knowledge_db.commit()
    row_id = knowledge_cur.lastrowid
    
    return row_id


def build_faiss_index():
    """构建FAISS索引"""
    global faiss_index
    knowledge_cur.execute("SELECT embedding FROM documents")
    embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in knowledge_cur.fetchall()]
    print("build_faiss_index: embeddings 数量 =", len(embeddings))
    if embeddings:
        print("embedding shape =", embeddings[0].shape, "dtype =", embeddings[0].dtype)
        faiss_index = faiss.IndexFlatL2(embeddings[0].shape[0])
        faiss_index.add(np.vstack(embeddings))
        print("faiss_index.ntotal =", faiss_index.ntotal)
    else:
        print("没有可用的 embedding，未构建索引")


def retrieve_knowledge(query: str, top_k: int = 100) -> List[str]:
    """通过FAISS检索相关知识"""
    query_embedding = embed_text(query).reshape(1, -1)
    print("retrieve_knowledge: faiss_index.ntotal =", faiss_index.ntotal)
    print("query_embedding shape =", query_embedding.shape, "dtype =", query_embedding.dtype)
    # 检查faiss_index是否有内容
    if faiss_index.ntotal == 0:
        print("faiss_index 为空，无法检索")
        return []
    distances, indices = faiss_index.search(query_embedding, top_k)
    print("检索到的 indices:", indices)
    print("检索到的 distances:", distances)
    # 新建连接和游标，避免跨线程问题
    conn = sqlite3.connect("knowledge_base.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT content FROM documents WHERE id IN ({})".format(
            ",".join(["?"] * len(indices[0]))
        ),
        indices[0].tolist()
    )
    results = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    print(results)
    return results


# ---------------------
# 记忆机制实现
# ---------------------
class ConversationMemory:
    """对话记忆管理"""

    def __init__(self, max_history_length: int = 10):
        self.history: Dict[str, List[Dict[str, Any]]] = {}  # 每个用户独立历史
        self.max_history_length = max_history_length
        self.user_profiles: Dict[str, Dict[str, Any]] = {}

    def add_message(self, user_id: str, role: str, content: str):
        """添加消息到对话历史，content必须为字符串"""
        if user_id not in self.history:
            self.history[user_id] = []
        self.history[user_id].append({"role": role, "content": content})
        # 保持历史长度
        if len(self.history[user_id]) > self.max_history_length:
            self.history[user_id].pop(0)

    def get_history(self, user_id: str) -> List[ChatCompletionMessageParam]:
        """获取符合OpenAI SDK类型要求的对话历史"""
        # 结合用户画像增强历史
        user_profile = self.user_profiles.get(user_id, {})

        # 创建符合类型要求的系统消息
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": f"用户画像: {json.dumps(user_profile)}"
        }

        # 转换历史消息为符合类型要求的格式
        formatted_history: List[ChatCompletionMessageParam] = []
        for message in self.history.get(user_id, []):
            if message["role"] == "system":
                formatted_message: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": message["content"]
                }
            elif message["role"] == "user":
                formatted_message: ChatCompletionUserMessageParam = {
                    "role": "user",
                    "content": message["content"]
                }
            elif message["role"] == "assistant":
                formatted_message: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": message["content"]
                }
            else:
                # 处理其他角色或作为回退
                formatted_message: ChatCompletionMessageParam = {
                    "role": message["role"],
                    "content": message["content"]
                }
            formatted_history.append(formatted_message)

        return [system_message] + formatted_history

    def update_profile(self, user_id: str, profile: Dict[str, Any]):
        """更新用户画像"""
        self.user_profiles[user_id] = {**self.user_profiles.get(user_id, {}), **profile}


memory = ConversationMemory()


# ---------------------
# 外部工具集成
# ---------------------
class ToolCaller:
    """外部工具调用器"""

    def call_calculator(self, expression: str) -> str:
        """调用计算器工具（支持常见数学表达式，安全性高于eval）"""
        try:
            # 支持常见数学函数和运算
            result = simple_eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误: {str(e)}"

    def query_database(self, query: str) -> str:
        """对本地 knowledge_base.db 数据库执行 SELECT 查询，并返回带表头的结果字符串"""
        try:
            if not query.strip().lower().startswith("select"):
                return "只允许 SELECT 查询。"
            conn = sqlite3.connect("knowledge_base.db")
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            if not rows:
                return "查询无结果。"
            # 获取列名
            col_names = [desc[0] for desc in cur.description]
            result_lines = [", ".join(col_names)]
            for row in rows:
                result_lines.append(", ".join(str(item) for item in row))
            return "\n".join(result_lines)
        except Exception as e:
            return f"数据库错误: {str(e)}"
        finally:
            try:
                cur.close()
                conn.close()
            except:
                pass

    def execute_python(self, code: str) -> str:
        """执行Python代码（有安全限制）"""
        # 只允许执行简单的表达式，不允许import、open等危险操作
        if "import" in code or "open" in code or "__" in code:
            return "不允许执行危险操作"
        try:
            # 只允许表达式求值
            result = eval(code, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"代码执行错误: {str(e)}"


tool_caller = ToolCaller()


def handle_tool_calls(response: str) -> str:
    """处理工具调用"""
    try:
        data = json.loads(response)
        if "tool" in data:
            tool_name = data["tool"]["name"]
            params = data["tool"]["parameters"]

            if tool_name == "calculator":
                result = tool_caller.call_calculator(params.get("expression", ""))
                return f"工具调用结果: {result}"
            elif tool_name == "python":
                result = tool_caller.execute_python(params.get("code", ""))
                return f"工具调用结果: {result}"
            elif tool_name == "database":
                result = tool_caller.query_database(params.get("query", ""))
                return f"工具调用结果: {result}"
            else:
                return f"未知工具: {tool_name}"
        return response
    except Exception as e:
        print(f"工具调用处理异常: {str(e)}")
        return response

def validate_input(input_data):
    """校验输入数据是否符合规范"""
    try:
        validate(instance=input_data, schema=input_schema)
        return True, None
    except ValidationError as e:
        return False, str(e)


def image_to_caption_blip2(image: Image.Image) -> str:
    """
    使用BLIP2模型生成图片内容描述
    """
    inputs = BLIP2_PROCESSOR(image, return_tensors="pt")
    out = BLIP2_MODEL.generate(**inputs)
    caption = BLIP2_PROCESSOR.decode(out[0], skip_special_tokens=True)
    return caption.strip() or "未能识别图片内容"


def audio_to_text_whisper(audio_file: str) -> str:
    """
    使用 Whisper 模型将音频文件转为文本
    """
    result = WHISPER_MODEL.transcribe(audio_file)
    return result["text"].strip() or "未能识别音频内容"


def generate_response(input_data, user_id="default_user"):
    if isinstance(input_data.get("prompt"), str):
        input_data["prompt"] = preprocess_input_text(input_data["prompt"])
        valid, error = validate_input(input_data)
        if not valid:
            return {"error": error}
    model = input_data.get("model", "deepseek-chat")
    temperature = input_data.get("temperature", 0.7)
    max_tokens = input_data.get("max_tokens", 2000)
    try:
        messages = memory.get_history(user_id)
        prompt = input_data["prompt"]
        # 高考志愿专用提示词
        if input_data.get("gaokao_mode", False):
            system_prompt = f"""
            你是一个高考志愿填报专家，请根据以下考生信息生成推荐方案：
            - 分数: {input_data.get('score', '未提供')}
            - 省份: {input_data.get('province', '未提供')}
            - 科类: {input_data.get('subject_type', '未提供')}
            
            输出必须严格符合以下JSON格式：
            {json.dumps(GAOKAO_OUTPUT_SCHEMA, indent=2, ensure_ascii=False)}
            
            要求：
            1. 必须包含candidate_analysis和recommendations
            2. 每个推荐必须包含university、major和probability
            3. probability只能是"冲刺"、"稳妥"或"保底"
            4. 每个推荐必须包含score_comparison字段，内容包括last_year_min（去年最低分）
            5. 必须包含strategy_advice字段，内容为对考生的填报策略建议
            """
            messages.insert(0, {"role": "system", "content": system_prompt})
        if isinstance(prompt, Image.Image):
            prompt_text = image_to_caption_blip2(prompt)
            messages.append({"role": "user", "content": f"图片内容描述：{prompt_text}"})
        elif isinstance(prompt, str) and prompt.startswith("audio:"):
            audio_file = prompt[6:]
            prompt_text = audio_to_text_whisper(audio_file)
            messages.append({"role": "user", "content": f"音频内容转写：{prompt_text}"})
        else:
            if input_data.get("use_rag", False):
                related_knowledge = retrieve_knowledge(input_data.get("rag_keyword", prompt))
                if related_knowledge:
                    messages.append({
                        "role": "system",
                        "content": "相关知识库信息:\n" + "\n---\n".join(related_knowledge)
                    })
            messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            response_format={"type": "json_object"} if input_data.get("gaokao_mode", False) else None
        )
        content = response.choices[0].message.content
        processed_content = handle_tool_calls(content)
        # 高考模式下验证输出结构，并自动补充分差
        if input_data.get("gaokao_mode", False):
            try:
                output_data = json.loads(processed_content)
                validate(instance=output_data, schema=GAOKAO_OUTPUT_SCHEMA)
                # 自动补充分差
                if "recommendations" in output_data:
                    for rec in output_data["recommendations"]:
                        if "score_comparison" in rec and "last_year_min" in rec["score_comparison"] and "score" in input_data:
                            rec["score_comparison"]["score_difference"] = input_data["score"] - rec["score_comparison"]["last_year_min"]
                processed_content = json.dumps(output_data, ensure_ascii=False)
            except Exception as e:
                print(f"高考志愿输出验证失败: {str(e)}")
                processed_content = enforce_structure(
                    f"分数: {input_data.get('score', '')}, 省份: {input_data.get('province', '')}, 科类: {input_data.get('subject_type', '')}",
                    GAOKAO_OUTPUT_SCHEMA
                )
        memory.add_message(user_id, "assistant", processed_content)
        tokens_used = getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0
        return {
            "response": processed_content,
            "model": model,
            "temperature": temperature,
            "tokens_used": tokens_used
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"网络错误: {str(e)}"}
    except (KeyError, json.JSONDecodeError) as e:
        return {"error": f"响应解析错误: {str(e)}"}
    except Exception as e:
        return {"error": f"未知错误: {str(e)}"}


def generate_response_stream(input_data, user_id="default_user"):
    """流式生成响应，逐步输出内容（打字机效果）"""
    if isinstance(input_data.get("prompt"), str):
        # 先做敏感词过滤和指令注入防护
        input_data["prompt"] = preprocess_input_text(input_data["prompt"])
        valid, error = validate_input(input_data)
        if not valid:
            yield f"输入校验错误: {error}"
            return

    model = input_data.get("model", "deepseek-chat")
    temperature = input_data.get("temperature", 0.7)
    max_tokens = input_data.get("max_tokens", 2000)

    try:
        messages = memory.get_history(user_id)
        prompt = input_data["prompt"]
        if isinstance(prompt, Image.Image):
            prompt_text = image_to_caption_blip2(prompt)
            messages.append({"role": "user", "content": f"图片内容描述：{prompt_text}"})
        elif isinstance(prompt, str) and prompt.startswith("audio:"):
            audio_file = prompt[6:]
            prompt_text = audio_to_text_whisper(audio_file)
            messages.append({"role": "user", "content": f"音频内容转写：{prompt_text}"})
        else:
            rag_keyword = input_data.get("rag_keyword", "").strip()
            if input_data.get("use_rag", False) and rag_keyword:
                related_knowledge = retrieve_knowledge(rag_keyword)
                if related_knowledge:
                    knowledge_context = "相关知识库信息:\n" + "\n---\n".join(related_knowledge)
                    messages.append({"role": "system", "content": knowledge_context})
            messages.append({"role": "user", "content": prompt})

        response_stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        full_content = ""
        for chunk in response_stream:
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                full_content += delta
                yield full_content
        # 更新对话记忆（流式只记录最终内容）
        if full_content:
            memory.add_message(user_id, "assistant", full_content)
    except Exception as e:
        yield f"流式响应错误: {str(e)}"


# 在这里插入 generate_gaokao_recommendation

def generate_gaokao_recommendation(input_data, user_id="default_user"):
    required_fields = ["score", "province", "subject_type"]
    for field in required_fields:
        if field not in input_data:
            return {"error": f"缺少必填字段: {field}"}
    # 自动生成 prompt 字段
    input_data["prompt"] = f"分数：{input_data['score']}，省份：{input_data['province']}，科类：{input_data['subject_type']}，兴趣：{','.join(input_data.get('interests', []))}"
    input_data["gaokao_mode"] = True
    input_data["use_rag"] = True  # 自动开启RAG
    input_data["rag_keyword"] = f"{input_data['score']} {input_data['province']} {input_data['subject_type']} {' '.join(input_data.get('interests', []))}"
    return generate_response(input_data, user_id)


# 创建Gradio界面（输出调整为3个文本框）
def create_web_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 高考志愿填报助手")

        # 顶部模式选择
        mode_select = gr.Radio(choices=["普通模式", "专家模式"], value="普通模式", label="请选择模式")

        # 普通模式区
        with gr.Column(visible=True) as normal_mode_area:
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(label="输入提示词", lines=5)
                    image_input = gr.Image(label="上传图片（可选）", type="pil")
                    audio_input = gr.Audio(label="上传音频（可选）", type="filepath")
                    model_choice = gr.Dropdown(
                        choices=["deepseek-chat", "deepseek-chat-large"],
                        label="选择DeepSeek模型",
                        value="deepseek-chat"
                    )
                    rag_checkbox = gr.Checkbox(label="开启RAG知识库检索", value=False)
                    stream_btn = gr.Button("流式输出")
            with gr.Row():
                
                stream_output = gr.Textbox(label="流式输出", lines=10)

        # 专家模式区
        with gr.Column(visible=False) as expert_mode_area:
            gr.Markdown("## 高考志愿专家模式")
            with gr.Row():
                with gr.Column():
                    gaokao_text_input = gr.Textbox(label="补充信息（可选）")
                    gaokao_audio_input = gr.Audio(label="上传语音（可选）", type="filepath")
                    gaokao_image_input = gr.Image(label="上传图片（可选）", type="pil")
                    score_input = gr.Number(label="分数", precision=0)
                    province_input = gr.Textbox(label="省份")
                    subject_type_input = gr.Dropdown(choices=["理科", "文科", "综合"], label="科类")
                    interests_input = gr.Textbox(label="兴趣（用逗号分隔）")
                    gaokao_smart_btn = gr.Button("智能推荐")
                with gr.Column():
                    gaokao_json_output = gr.JSON(label="推荐结果（结构化JSON）")
                    gaokao_analysis_output = gr.Textbox(label="考生分析", lines=5)
                    gaokao_strategy_output = gr.Textbox(label="填报策略建议", lines=5)
                    gaokao_stream_output = gr.Textbox(label="流式分析原文", lines=10)
            with gr.Row():
                gaokao_rec_table = gr.Dataframe(
                    headers=["院校", "专业", "录取概率", "去年最低分", "分差"],
                    label="推荐院校专业列表"
                )

        def multimodal_stream(prompt, image, audio, model, use_rag):
            if image is not None:
                input_prompt = image
            elif audio is not None:
                input_prompt = f"audio:{audio}"
            else:
                input_prompt = prompt
            input_data = {
                "prompt": input_prompt,
                "model": model,
                "temperature": 0.7,
                "use_rag": use_rag,
                "rag_keyword": prompt
            }
            for chunk in generate_response_stream(input_data, user_id="default_user"):
                yield chunk

        def gaokao_smart_callback(text, audio, image, score, province, subject_type, interests):
            import time
            # 0. 先清空所有输出区
            yield {
                gaokao_stream_output: "",
                gaokao_json_output: {},
                gaokao_analysis_output: "",
                gaokao_strategy_output: "",
                gaokao_rec_table: []
            }
            # 1. 组装 prompt
            prompt_parts = []
            if text:
                prompt_parts.append(f"补充信息：{text}")
            if audio:
                audio_text = audio_to_text_whisper(audio)
                prompt_parts.append(f"语音内容：{audio_text}")
            if image is not None:
                image_text = image_to_caption_blip2(image)
                prompt_parts.append(f"图片内容：{image_text}")
            prompt = " ".join(prompt_parts)
            input_data = {
                "prompt": f"分数：{score}，省份：{province}，科类：{subject_type}，兴趣：{interests}。{prompt}。请推荐不超过10个院校。",
                "model": "deepseek-chat",
                "temperature": 0.7,
                "use_rag": True,
                "rag_keyword": f"{score} {province} {subject_type} {interests}"
            }
            # 2. 流式分析输出，保留秒数显示
            full_content = ""
            start_time = time.time()
            for chunk in generate_response_stream(input_data, user_id="default_user"):
                full_content = chunk
                elapsed = time.time() - start_time
                display_content = f"{full_content}\n\n已用时：{elapsed:.1f}秒"
                yield {
                    gaokao_stream_output: display_content,
                    gaokao_json_output: {},
                    gaokao_analysis_output: "",
                    gaokao_strategy_output: "",
                    gaokao_rec_table: []
                }
            # 3. 自动结构化
            struct_prompt = f"请将以下内容结构化为指定JSON格式：\n{full_content}\n\n格式要求：{json.dumps(GAOKAO_OUTPUT_SCHEMA, ensure_ascii=False)}"
            struct_input_data = {
                "prompt": struct_prompt,
                "model": "deepseek-chat",
                "gaokao_mode": True,
                "score": score,
                "province": province,
                "subject_type": subject_type,
                "interests": [i.strip() for i in interests.split(",") if i.strip()]
            }
            result = generate_response(struct_input_data)
            try:
                data = json.loads(result["response"])
                candidate_analysis = data.get("candidate_analysis", "")
                recommendations = data.get("recommendations", [])
                strategy_advice = data.get("strategy_advice", "")
                table_data = []
                for rec in recommendations:
                    last_year_min = rec.get("score_comparison", {}).get("last_year_min", "")
                    score_diff = rec.get("score_comparison", {}).get("score_difference", "")
                    table_data.append([
                        rec.get("university", ""),
                        rec.get("major", ""),
                        rec.get("probability", ""),
                        last_year_min,
                        score_diff
                    ])
                analysis_text = candidate_analysis if isinstance(candidate_analysis, str) else json.dumps(candidate_analysis, ensure_ascii=False)
                yield {
                    gaokao_stream_output: f"{full_content}\n\n已用时：{elapsed:.1f}秒",
                    gaokao_json_output: data,
                    gaokao_analysis_output: analysis_text,
                    gaokao_strategy_output: strategy_advice,
                    gaokao_rec_table: table_data
                }
            except Exception as e:
                yield {
                    gaokao_stream_output: f"{full_content}\n\n已用时：{elapsed:.1f}秒",
                    gaokao_json_output: {"error": str(e)},
                    gaokao_analysis_output: "解析失败",
                    gaokao_strategy_output: "解析失败",
                    gaokao_rec_table: []
                }

        # 模式切换回调
        def switch_mode(mode):
            return (
                gr.update(visible=(mode == "普通模式")),
                gr.update(visible=(mode == "专家模式"))
            )

        mode_select.change(
            fn=switch_mode,
            inputs=mode_select,
            outputs=[normal_mode_area, expert_mode_area]
        )

        stream_btn.click(
            fn=multimodal_stream,
            inputs=[prompt_input, image_input, audio_input, model_choice, rag_checkbox],
            outputs=stream_output,
            api_name=None,
            queue=True
        )
        gaokao_smart_btn.click(
            fn=gaokao_smart_callback,
            inputs=[gaokao_text_input, gaokao_audio_input, gaokao_image_input, score_input, province_input, subject_type_input, interests_input],
            outputs=[gaokao_stream_output, gaokao_json_output, gaokao_analysis_output, gaokao_strategy_output, gaokao_rec_table]
        )

    return interface


# 运行界面
if __name__ == "__main__":
    build_faiss_index()
    interface = create_web_interface()
    interface.launch(server_port=7861, share=False, show_error=True, inbrowser=True)
