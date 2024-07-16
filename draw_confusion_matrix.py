import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main():
    y_pred = np.load('all_pred.npy')
    y_true = np.load('all_labels.npy')
    
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(15, 15))
    class_labels = [
    "drink water",
    "eat meal",
    "brush teeth",
    "brush hair",
    "drop",
    "pick up",
    "throw",
    "sit down",
    "stand up",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "put on jacket",
    "take off jacket",
    "put on a shoe",
    "take off a shoe",
    "put on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping",
    "jump up",
    "phone call",
    "play with phone/tablet",
    "type on a keyboard",
    "point to something",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put palms together",
    "cross hands in front",
    "sneeze/cough",
    "staggering",
    "falling down",
    "headache",
    "chest pain",
    "back pain",
    "neck pain",
    "nausea/vomiting",
    "fan self",
    "punch/slap",
    "kicking",
    "pushing",
    "pat on back",
    "point finger",
    "hugging",
    "giving object",
    "touch pocket",
    "shaking hands",
    "walking towards",
    "walking apart"
    ]


    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    plt.xticks(np.arange(len(class_labels)), class_labels, rotation=90)
    plt.yticks(np.arange(len(class_labels)), class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.savefig('confusion_matrix.png', dpi=800)
    plt.show()


if __name__ == "__main__":
    main()