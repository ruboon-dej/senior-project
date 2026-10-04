#!/usr/bin/env python
# coding: utf-8


# Import libraries
import streamlit as st
import tensorflow as tf
import pathlib
import random
import string
import re
import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

import keras as keras2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from spoken_to_signed.gloss_to_pose.concatenate import concatenate_poses, reduce_holistic, normalize_pose, smooth_concatenate_poses, correct_wrists, trim_pose
from typing import List
from pose_format import Pose
from pose_format.utils.holistic import BODY_POINTS
from pose_format.pose_visualizer import PoseVisualizer

from pythainlp import word_tokenize

from transformer_components import TransformerEncoder, TransformerDecoder, PositionalEmbedding
from modified_trim_pose import list_components

np.set_printoptions(suppress=False)
pd.set_option('display.max_rows', 500)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(tf.config.list_physical_devices('GPU'))

# Load dataset
df = pd.read_csv("datasets/deepcut.csv") 
with open("datasets/th_vocab-deepcut.txt", 'r', encoding="utf8") as f:
    data = f.read()
    word_bag = data.split('\n')
print(len(word_bag))
 
text_pairs = df.values.tolist()

for _ in range(5):
    print(random.choice(text_pairs))

# Create train, validate, test set
random.Random(10).shuffle(text_pairs)
num_val_samples = int(0.1 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")


# Prepare dataset for training
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = len(word_bag)
sequence_length = 20
batch_size = 64


def custom_standardization(input_string):
    # lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(input_string, "[%s]" % re.escape(strip_chars), "")


thai_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
thsl_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_thai_texts = [pair[0] for pair in train_pairs]
train_thsl_texts = [pair[1] for pair in train_pairs]
thai_vectorization.adapt(train_thai_texts)
thsl_vectorization.adapt(train_thsl_texts)

def format_dataset(thai, thsl):
    thai = thai_vectorization(thai)
    thsl = thsl_vectorization(thsl)
    return (
        {
            "encoder_inputs": thai,
            "decoder_inputs": thsl[:, :-1],
        },
        thsl[:, 1:],
    )

def make_dataset(pairs):
    thai_texts, thsl_texts = zip(*pairs)
    thai_texts = list(thai_texts)
    thsl_texts = list(thsl_texts)
    dataset = tf.data.Dataset.from_tensor_slices((thai_texts, thsl_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)


# Load Pretrained model
custom_objects = {"PositionalEmbedding": PositionalEmbedding,
                    "TransformerEncoder": TransformerEncoder,
                  "TransformerDecoder":TransformerDecoder,
                 }

transformer = tf.keras.models.load_model('models/thai2thsl-deepcut.hdf5', custom_objects=custom_objects)

# Show the model architecture
transformer.summary()


# Decode Sequence
thsl_vocab = thsl_vectorization.get_vocabulary()
thsl_index_lookup = dict(zip(range(len(thsl_vocab)), thsl_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = thai_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = thsl_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = thsl_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence.replace("[start]", "").replace("[end]", "").strip()


# Create a binary classification model
model = Sequential()
model.add(Dense(120, input_shape=(120,), activation="relu"))
model.add(Dense(60, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()

# Load Pretrained model
model = tf.keras.models.load_model('models/pose-detector.hdf5')
pose_directory = 'e-tsl-pose'

def thai_to_tsl_glosses(sent: str) -> List:
    """
    This function converts Thai text to glosses.

    Parameters:
        :param str sent: a Thai sentence such as ฉันกินส้มตำ, พ่อไปตลาด, แม่อยู่ที่บ้านตอนเช้า.

    Returns: 
        :return: a list of glosses.
    """
    s1 = word_tokenize(sent)
    if len(s1) == 1:
        return s1
    s = " ".join(s1)
    if re.search('กินข้าว', s):
        find_eat = re.search('กิน', s)
        s = s[:find_eat.end()] + " " + s[find_eat.end():]
    glosses = decode_sequence(s)
    return glosses.split(" ")  

def merge_multiword_glosses(glosses: List[str], available: set) -> List[str]:
    merged = []
    i = 0
    while i < len(glosses):
        if i + 1 < len(glosses):
            first, second = glosses[i], glosses[i + 1]
            candidate = next(
                (c for c in (f"{first} {second}", f"{first} ({second})") if c in available),
                None,
            )
            if candidate is not None:
                merged.append(candidate)
                i += 2
                continue
        merged.append(glosses[i])
        i += 1
    return merged

def lookup(glosses: List[str]):
    """
    This function load the pose data of the corresponding to the gloss from the file system.

    Parameters:
        :param List[str] glosses: a list of glosses such as ['ฉัน', 'กิน', 'ส้มตำ'], [พ่อ, ไป, ตลาด], [แม่, อยู่, ที่, บ้าน, ตอนเช้า].

    Returns: 
        :return: a list of found pose data.
        :rtype: List[str]
        :return: a list of glosses that doesn't have the corresponding pose data.
        :rtype: List[str]
    """
    # Some glosses in the training data are written as "A / B" synonym
    # pairs (e.g. "คอมพิวเตอร์ / สมองกล"), but the corresponding pose
    # file is stored under neither Thai name — only under a third,
    # English name. Map every known synonym (both sides of the pair)
    # to the pose file name that actually exists on disk.
    gloss_synonyms = {
        "คอมพิวเตอร์": "computer",
        "สมองกล": "computer",
        "ที่ว่าการอำเภอ": "district_office",
        "สำนักงานเขต": "district_office",
    }

    pose_list = []
    not_found_list = []
    available_poses = {
        f[:-5] for f in os.listdir(os.path.join(os.getcwd(), pose_directory)) if f.endswith(".pose")
    }
    glosses = merge_multiword_glosses(list(glosses), available_poses)
    for gloss in glosses:
        pattern = r'^(\D+)([0-9])$'
        file_name = gloss
        if not file_name.isnumeric():
            file_name = re.sub(pattern, r'\1(\2)', file_name)
        file_name = "{}.pose".format(file_name)
        pose_path = os.path.join(os.getcwd(), pose_directory, file_name)

        if not os.path.exists(pose_path):
            # Fallback 1: some glosses are stored as "gloss(1).pose",
            # "gloss(2).pose", etc. instead of a plain "gloss.pose".
            # Try the lowest-numbered variant before giving up.
            import glob
            variant_pattern = os.path.join(os.getcwd(), pose_directory, "{}([0-9]*).pose".format(gloss))
            variants = sorted(glob.glob(variant_pattern))
            if variants:
                pose_path = variants[0]

        if not os.path.exists(pose_path) and gloss in gloss_synonyms:
            # Fallback 2: known synonym pairs whose pose file uses a
            # different name entirely (see gloss_synonyms above).
            synonym_file_name = "{}.pose".format(gloss_synonyms[gloss])
            synonym_path = os.path.join(os.getcwd(), pose_directory, synonym_file_name)
            if os.path.exists(synonym_path):
                pose_path = synonym_path

        try:
            with open(pose_path, "rb") as f:
                pose = Pose.read(f.read())
                print(f"{gloss} {pose.body.fps} {len(pose.body.data)}")
                # pose.body.data = pose.body.data[30:-60]
                # pose.body.confidence= pose.body.confidence[30:-60]
                pose_list.append(pose)
        except Exception as err:
            print(err)
            not_found_list.append(gloss)
    return pose_list, not_found_list

# test1 = ["มะละกอ", "หวาน", "แม่", "กิน", "พ่อ", "เบิกเงิน", "ทหาร", "เขย่า", "ทารก", "ข้าว", "ไม่อร่อย", "โรงพยาบาล"]
# test2 = ["หมา", "ทุเรียน", "ไทย"]
# test3 = ["คนหูดี", "แมว", "พังทลาย", "ทำงาน", "เดิน"]
# pose_list = lookup(test1)
# df, new_pose = list_components(pose_list[4], True)
# df.head()


import itertools,operator
np.set_printoptions(suppress=False)
# https://stackoverflow.com/a/40166998
def thsl_trim_pose(pose, model: keras2.src.engine.sequential.Sequential, 
                   start=True, end=True, frame_padding=5) -> Pose:
    """
    This function trims out the waiting pose frames.

    Parameters:
        :param Pose pose: a pose file to be trimmed
        :param keras2.src.engine.sequential.Sequential model: the binary classifier model
        :param bool start: Is it a beginning pose in a sentence?
        :param bool end: Is it an ending pose in a sentence?
        :param int frame_padding: number of frames to be padded

    Returns: 
        :return: a trimmed pose
        :rtype: Pose
    """
    
    if len(pose.body.data)-1 == 0:
        return pose

    df, _ = list_components(pose, False)
    df_list = df.values.tolist()
    
    start_index = 0
    end_index = len(pose.body.data)-1

    data = model.predict(df_list)
    data = np.array([1 if x >= 0.5 else 0 for x in data])
    
    r = max((list(y) for (x,y) in itertools.groupby((enumerate(data)),operator.itemgetter(1)) if x == 0), key=len)
    if start == True:
        start_index = r[0][0] + frame_padding
    if end == True:
        end_index = r[-1][0] - frame_padding

    minimum_require = np.floor(0.95 * len(pose.body.data))
    if (end_index - start_index) > minimum_require:
        pose = trim_pose(pose)
        return pose

    print(start_index, end_index, start, end)
    pose.body.data = pose.body.data[start_index:end_index]
    pose.body.confidence = pose.body.confidence[start_index:end_index]
    return pose


def modified_concatenate_poses(poses: List[Pose], model: keras2.src.engine.sequential.Sequential) -> Pose:
    """
    This function concatenates the pose together, and is integrated with the new trim pose function

    Parameters:
        :param List[Pose] poses: a list of pose data to concatenate them together
        :param keras2.src.engine.sequential.Sequential model: the binary classifier model

    Returns: 
        :return: concatenated pose
        :rtype: Pose
    """
    print('Reducing poses...')
    poses = [reduce_holistic(p) for p in poses]

    print('Normalizing poses...')
    poses = [normalize_pose(p) for p in poses]

    # Trim the poses to only include the parts where the pose is the gloss
    print('Trimming poses...')
    poses = [thsl_trim_pose(p, model, i > 0, i < len(poses) - 1) for i, p in enumerate(poses)]

    # Concatenate all poses
    print('Smooth concatenating poses...')
    pose = smooth_concatenate_poses(poses)

    # Correct the wrists (should be after smoothing)
    print('Correcting wrists...')
    pose = correct_wrists(pose)

    # Scale the newly created pose
    print('Scaling pose...')
    new_width = 500
    shift = 1.25
    shift_vec = np.full(shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
    pose.body.data = (pose.body.data + shift_vec) * new_width
    pose.header.dimensions.height = pose.header.dimensions.width = int(new_width * shift * 2)

    return pose

def pose_sequence(pose_list):
    """
    This function export the pose data into a .mp4 video.

    Parameters:
        :param List[Pose] pose_list: a list of poses to concatenate together
        :param keras2.src.engine.sequential.Sequential model: the binary classifier model

    Returns: 
        None
    """
    
    # p = pose_list
    p = modified_concatenate_poses(pose_list, model)
    # p.body.interpolate(60, kind='cubic')
    scale = p.header.dimensions.width / 256
    p.header.dimensions.width = int(p.header.dimensions.width / scale)
    p.header.dimensions.height = int(p.header.dimensions.height / scale)
    p.body.data = p.body.data / scale

    # Genearate .gif
    v = PoseVisualizer(p)
    
    v.save_video("quick_test.mp4", v.draw())

print("Thai2ThSL loaded")

# test_pose_list = lookup(["ฉัน", "เห็น", "คุณ", "โรงเรียน"])
# test_pose_list = lookup(["ฉัน", "โรงเรียน", "เดิน", "ไป"])
# pose_sequence(test_pose_list)

# sent = 'ฉันนอนที่บ้าน'
# glosses = thai_to_tsl_glosses(sent)
# print(glosses)
# pose_list = lookup(glosses)
# pose_sequence(pose_list)