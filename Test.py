from llm_axe.models import OllamaChat
from llm_axe import ObjectDetectorAgent

llm = OllamaChat(model="llava:7b")
detector = ObjectDetectorAgent(llm, llm)

resp = detector.detect(images=["Water-Barrels.jpg"], detection_criteria="Things related to water")
print(resp)