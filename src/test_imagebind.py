import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


import sys
sys.path.append('./src/imagebind')

if __name__ == '__main__':
    text_list = ["A very beautiful canine.", "A car", "A bird", "A fat cartoon rabbit leaving his home", "a fat animal with big ears is gray"]
    image_paths = ["samples/dog_image.jpg", "samples/car_image.jpg", "samples/bird_image.jpg"]
    audio_paths = ["samples/dog_audio.wav", "samples/car_audio.wav", "samples/bird_audio.wav"]
    video_paths = ["samples/rabbit_cartoon.mp4"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(ckpt_path='/home/adryw/dataset/imagecraft/imagebind_huge.pth', pretrained=True)
    model.eval()
    model.to(device)

    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    print(
        "Vision x Text: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )
    print(
        "Audio x Text: ",
        torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )
    print(
        "Vision x Audio: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
    )
    print(
        "Video x Text: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )

    # Test with video
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.VISION: data.load_and_transform_video_data(video_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    print(
        "Vision x Text: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )