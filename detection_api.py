import io, os
from numpy import random
from google.cloud import vision_v1
from pillow_utility import draw_borders, Image
import pandas as pd
from google.cloud.vision_v1 import types



os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"property-detection.json"
client = vision_v1.ImageAnnotatorClient()

file_name = 'team3.jpg'
image_path = os.path.join('.\Images', file_name)

with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

image = vision_v1.types.Image(content=content)
response = client.object_localization(image=image)
localized_object_annotations = response.localized_object_annotations

# crop_hints_params = vision_v1.types.CropHintsParams(aspect_ratios=aspect_ratios)

pillow_image = Image.open(image_path)
df = pd.DataFrame(columns=['name', 'score'])
for obj in localized_object_annotations:
    df = df.append(
        dict(
            name=obj.name,
            score=obj.score
        ),
        ignore_index=True)

    r, g, b = random.randint(150, 255), random.randint(
        150, 255), random.randint(150, 255)

    draw_borders(pillow_image, obj.bounding_poly, (r, g, b),
                 pillow_image.size, obj.name, obj.score)

print(df)
pillow_image.show()
