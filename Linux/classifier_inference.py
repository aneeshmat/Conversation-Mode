import torch
import numpy as np
from model import VoiceClassifier
from feature_extractor import extract_features

clf = VoiceClassifier(input_dim=18)
clf.load_state_dict(torch.load("voice_classifier.pt"))
clf.eval()

def classifier_prob(cleaned_dev, ref_dev):
    feats = extract_features(cleaned_dev, ref_dev)
    with torch.no_grad():
        p = clf(torch.from_numpy(feats).unsqueeze(0))
    return float(p.item())

