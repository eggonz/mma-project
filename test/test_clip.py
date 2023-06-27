from clip import load_clip_model, compute_emb_distances
from dataset import FundaPrecomputedEmbeddings, FundaDataset


def test_compute_emb_distances():
    """
    integration test for compute_emb_distances
    """
    fpe = FundaPrecomputedEmbeddings.from_pickle('../data/clip_embeddings/funda_images_tiny.pkl')
    fds = FundaDataset.from_jsonlines('../data/Funda/Funda/ads.jsonlines', '../data/funda_images_tiny')
    model = load_clip_model()

    doc_embs = fpe.get_all_embeddings()
    imgs = fds.get_images(42194092)
    query_img = list(imgs.values())[0]
    query_emb = model.get_visual_emb_for_img(query_img)

    distances = compute_emb_distances(query_emb, doc_embs)
    print(distances)


if __name__ == '__main__':
    # run tests as script
    test_compute_emb_distances()
