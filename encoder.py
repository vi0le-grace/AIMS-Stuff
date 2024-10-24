import numpy as np
import pandas as pd

#Ordinal Encoder
def ordinal_encoder(df, column, order):

    ordinal_mapping = {category: rank for rank, category in enumerate(order, 1)}
    df[column + '_Ordinal'] = df[column].map(ordinal_mapping)
    
    return df

#Onehot Encoder
def onehot_encoder(df, column):

    one_hot = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, one_hot], axis=1)
    
    return df

data = {
    'Color': ['Red','Green','Blue','Blue','Blue','Red'],
    'Height': ['short','tall','dwarf','average','average','tall']
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

height_order = ['dwarf','short','average','tall']
df = ordinal_encoder(df, 'Height', height_order)
print("\nAfter ordinal encoding:\n", df)

df = onehot_encoder(df, 'Color')
print("\nAfter one hot encoding:\n", df)