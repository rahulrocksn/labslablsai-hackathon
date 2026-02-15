import mediapipe
print(dir(mediapipe))
try:
    print(mediapipe.solutions)
except AttributeError:
    print("FAILED: solutions not found")
