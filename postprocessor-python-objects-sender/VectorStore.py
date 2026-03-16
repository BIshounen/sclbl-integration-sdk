import uuid
import torch
import open_clip
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class VectorStore:
  def __init__(self, qdrant_url="http://localhost:6333", collection_name="objects", api_token=None):
    # Load OpenCLIP model
    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    self.model.eval()

    # Connect to Qdrant
    self.client = QdrantClient(url=qdrant_url, api_key=api_token)
    self.collection_name = collection_name
    self._ensure_collection()

  def _ensure_collection(self):
    collections = [c.name for c in self.client.get_collections().collections]
    if self.collection_name not in collections:
      self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
      )

  def vectorize(self, image_bgr: np.ndarray) -> list[float]:
    """Takes a BGR numpy image (from cv2), returns embedding."""
    image_rgb = Image.fromarray(image_bgr[:, :, ::-1])
    image_tensor = self.preprocess(image_rgb).unsqueeze(0)

    with torch.no_grad():
      embedding = self.model.encode_image(image_tensor)
      embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding[0].cpu().numpy().tolist()

  def store(self, image_bgr: np.ndarray, metadata: dict = None) -> str:
    """Vectorize and store in Qdrant. Returns the point ID."""
    embedding = self.vectorize(image_bgr)
    point_id = str(uuid.uuid4())

    self.client.upsert(
      collection_name=self.collection_name,
      points=[
        PointStruct(
          id=point_id,
          vector=embedding,
          payload=metadata or {},
        )
      ],
    )
    return point_id

  def search(self, image_bgr: np.ndarray, limit: int = 5):
    """Search for similar images."""
    embedding = self.vectorize(image_bgr)
    return self.client.query_points(
      collection_name=self.collection_name,
      query=embedding,
      limit=limit,
    )