from llm_axe import OllamaChat
from llm_axe import ObjectDetectorAgent

llm = OllamaChat(model="llava:latest")
detector = ObjectDetectorAgent(llm, llm, 0.3, 0.3, False)

resp = detector.detect(images=["./Water-Barrels.jpg"], detection_criteria="Things related to water and the wall behind them")
print(resp)