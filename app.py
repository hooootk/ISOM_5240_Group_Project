import streamlit as st
import torch
import warnings
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

warnings.filterwarnings('ignore')

st.set_page_config(page_title="ABCDEFG", page_icon="🛡️")
st.title("ABCDEFG")

# 模型路径
MODEL_B_PATH = "./distilbert-base-uncased-beavertails-final"

# 检查模型B是否存在
if not os.path.exists(MODEL_B_PATH):
    st.error(f"模型不存在: {MODEL_B_PATH}")
    st.stop()


@st.cache_resource
def load_model_a():
    """加载文本生成模型 (OPT-350M)"""
    try:
        generator = pipeline(
            "text-generation",
            model="facebook/opt-350m",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return generator
    except Exception as e:
        st.error(f"加载模型A失败: {e}")
        return None


@st.cache_resource
def load_model_b():
    """加载安全分类模型 (DistilBERT)"""
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


def generate_content(generator, prompt):
    """使用 OPT-350M 生成内容"""
    try:
        result = generator(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
        
        generated_text = result[0]['generated_text']
        
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text, None
    except Exception as e:
        return None, str(e)


def check_safety(tokenizer, model, device, prompt, response):
    """使用 DistilBERT 检测内容是否安全"""
    try:
        text = f"{prompt} [SEP] {response}"
        
        inputs = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False  # ✅ 关键修复：不返回 token_type_ids
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()
        
        is_safe = (pred == 0)
        return is_safe, confidence, None
        
    except Exception as e:
        return False, 0.0, str(e)


# 加载模型
with st.spinner("加载模型中..."):
    generator = load_model_a()
    tokenizer, model_b, device = load_model_b()

if generator is None or tokenizer is None or model_b is None:
    st.stop()

# 界面
prompt = st.text_input("请输入提示词：", placeholder="例如：写一篇关于人工智能的文章")

if st.button("生成"):
    if not prompt:
        st.warning("请输入提示词")
    else:
        with st.spinner("生成中..."):
            generated_content, gen_error = generate_content(generator, prompt)
        
        if gen_error:
            st.error(gen_error)
        else:
            with st.spinner("检测中..."):
                is_safe, confidence, safety_error = check_safety(
                    tokenizer, model_b, device, prompt, generated_content
                )
            
            if safety_error:
                st.error(safety_error)
            else:
                st.markdown("---")
                st.subheader("生成结果")
                st.write(generated_content)
                
                st.subheader("安全检测")
                if is_safe:
                    st.write(f"结果: Safe")
                    st.write(f"置信度: {confidence:.4f}")
                else:
                    st.write(f"结果: Unsafe")
                    st.write(f"置信度: {confidence:.4f}")
                    st.warning("⚠️ 内容不安全")
