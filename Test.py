from llm_axe import OllamaChat, ObjectDetectorAgent

# --- 1. First LLM: Object detection (LLava) ---
detector_llm = OllamaChat(model="llava:latest")
detector = ObjectDetectorAgent(detector_llm, detector_llm, 0.3, 0.3, False)

resp = detector.detect(
    images=["./Water-Barrels.jpg"],
    detection_criteria="Things related to water and the wall behind them"
)

print("RAW JSON:")
print(resp)

# --- 2. Second LLM: Analysis (GPT-OSS or any model you want) ---
analyzer_llm = OllamaChat(model="gpt-oss:latest")

prompt = f"""
Summarize this object detection JSON and describe the scene in simple language:

{resp}
"""

print(dir(analyzer_llm))
summary = analyzer_llm.ask(prompt)
print("\nSUMMARY:")
print(summary)
