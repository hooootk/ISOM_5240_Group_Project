"""
Streamlit 应用：AI 内容生成与安全检测系统
标题：ABCDEFG
"""

import streamlit as st
import torch
import warnings
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

# 忽略警告
warnings.filterwarnings('ignore')

# ============================================
# 页面配置
# ============================================
st.set_page_config(
    page_title="ABCDEFG",
    page_icon="🛡️"
)

# ============================================
# 标题
# ============================================
st.title("ABCDEFG")

# ============================================
# 配置模型路径
# ============================================
# 模型A：文本生成模型 (OPT-350M)
MODEL_A_NAME = "facebook/opt-350m"

# 模型B：安全分类模型（本地路径）
MODEL_B_PATH = "./distilbert-base-uncased-beavertails-final"

# 检查模型B是否存在
if not os.path.exists(MODEL_B_PATH):
    st.error(f"模型不存在: {MODEL_B_PATH}")
    st.stop()


# ============================================
# 加载模型
# ============================================
@st.cache_resource
def load_model_a():
    """加载文本生成模型 (OPT-350M)"""
    try:
        # 使用 OPT-350M 模型
        generator = pipeline(
            "text-generation",
            model=MODEL_A_NAME,
            device_map="auto",  # 自动分配到 GPU（如果有）
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return generator
    except Exception as e:
        st.error(f"加载模型A失败: {e}")
        return None


@st.cache_resource
def load_model_b():
    """加载安全分类模型"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_B_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_B_PATH)
        model = model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"加载模型B失败: {e}")
        return None, None, None


# ============================================
# 生成内容函数
# ============================================
def generate_content(generator, prompt):
    """使用 OPT-350M 生成内容"""
    try:
        # OPT-350M 生成参数
        result = generator(
            prompt,
            max_new_tokens=200,  # 生成的最大新 token 数
            do_sample=True,       # 使用采样
            temperature=0.7,      # 控制随机性
            top_p=0.9,            # nucleus sampling
            num_return_sequences=1
        )
        
        # 提取生成的文本
        generated_text = result[0]['generated_text']
        
        # 如果生成的内容包含原始提示词，只取新生成的部分
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text, None
    except Exception as e:
        return None, str(e)


# ============================================
# 安全检测函数
# ============================================
def check_safety(tokenizer, model, device, prompt, response):
    """使用模型B检测内容是否安全"""
    try:
        text = f"{prompt} [SEP] {response}"
        
        inputs = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()
        
        # pred=0 表示 safe，pred=1 表示 unsafe
        is_safe = (pred == 0)
        return is_safe, confidence, None
        
    except Exception as e:
        return False, 0.0, str(e)


# ============================================
# 加载模型
# ============================================
with st.spinner("加载模型中..."):
    generator = load_model_a()
    tokenizer, model_b, device = load_model_b()

if generator is None or tokenizer is None or model_b is None:
    st.stop()

# ============================================
# 界面
# ============================================
# 输入框
prompt = st.text_input("请输入提示词：", placeholder="例如：写一篇关于人工智能的文章")

# 生成按钮
if st.button("生成"):
    if not prompt:
        st.warning("请输入提示词")
    else:
        # 生成内容
        with st.spinner("生成中..."):
            generated_content, gen_error = generate_content(generator, prompt)
        
        if gen_error:
            st.error(gen_error)
        else:
            # 安全检测
            with st.spinner("检测中..."):
                is_safe, confidence, safety_error = check_safety(
                    tokenizer, model_b, device, prompt, generated_content
                )
            
            if safety_error:
                st.error(safety_error)
            else:
                # 显示生成的内容
                st.markdown("---")
                st.subheader("生成结果")
                st.write(generated_content)
                
                # 显示检测结果
                st.subheader("安全检测")
                if is_safe:
                    st.write(f"结果: Safe")
                    st.write(f"置信度: {confidence:.4f}")
                else:
                    st.write(f"结果: Unsafe")
                    st.write(f"置信度: {confidence:.4f}")
                    st.warning("⚠️ 内容不安全")
