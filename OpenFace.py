import cv2
import openface

# Load the pre-trained models
dlib_model_path = "path/to/dlib/shape_predictor_68_face_landmarks.dat"
align = openface.AlignDlib(dlib_model_path)

# Load an image
image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)

# Detect face bounding box
rectangles = align.getAllFaceBoundingBoxes(image)

# Iterate through detected faces
for rect in rectangles:
    landmarks = align.findLandmarks(image, rect)
    
    # Draw facial landmarks on the image
    for point in landmarks:
        cv2.circle(image, point, 2, (0, 0, 255), -1)

# Display the image with facial landmarks
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
