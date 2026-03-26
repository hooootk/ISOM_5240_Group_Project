import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

st.set_page_config(page_title="ABCDEFG", page_icon="🛡️")
st.title("ABCDEFG")

MODEL_B_PATH = "./distilbert-base-uncased-beavertails-final"

if not os.path.exists(MODEL_B_PATH):
    st.error(f"Model not found: {MODEL_B_PATH}")
    st.stop()


@st.cache_resource
def load_models():
    generator = pipeline("text-generation", model="facebook/opt-350m", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_B_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_B_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return generator, tokenizer, model, device


# Initialize session state
if 'generated' not in st.session_state:
    st.session_state.generated = None
if 'is_safe' not in st.session_state:
    st.session_state.is_safe = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'show_blocked' not in st.session_state:
    st.session_state.show_blocked = False
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = None


def clear_output():
    st.session_state.generated = None
    st.session_state.is_safe = None
    st.session_state.confidence = None
    st.session_state.show_blocked = False


generator, tokenizer, model_b, device = load_models()

prompt = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        clear_output()
        st.session_state.current_prompt = prompt
        
        # Generate
        result = generator(
            prompt,
            max_new_tokens=150,
            return_full_text=False,
            do_sample=True,
            temperature=0.6
        )
        generated = result[0]['generated_text'].strip()
        
        # Detect
        text = f"{prompt} [SEP] {generated}"
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", return_token_type_ids=False).to(device)
        
        with torch.no_grad():
            outputs = model_b(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs).item()
            confidence = probs[0][pred].item()
        
        st.session_state.generated = generated
        st.session_state.is_safe = (pred == 0)
        st.session_state.confidence = confidence
        
        st.rerun()

# Display results
if st.session_state.generated is not None:
    st.markdown("---")
    st.subheader("Safety Detection")
    
    if st.session_state.is_safe:
        st.success(f"✅ Safe (Confidence: {st.session_state.confidence:.4f})")
        st.subheader("Generated Content")
        st.write(st.session_state.generated)
    else:
        st.error(f"❌ Unsafe (Confidence: {st.session_state.confidence:.4f})")
        st.warning("⚠️ Content has been flagged as unsafe and is hidden.")
        
        if st.button("📋 Reveal Blocked Content"):
            st.session_state.show_blocked = True
            st.rerun()
        
        if st.session_state.show_blocked:
            st.subheader("Blocked Content")
            st.write(st.session_state.generated)
