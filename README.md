# Transfer Learning

## O que é Transfer Learning?

Transfer Learning é uma técnica de aprendizado de máquina onde um modelo pré-treinado em uma tarefa (normalmente com um grande conjunto de dados) é reutilizado para uma tarefa diferente, mas relacionada. O objetivo é aproveitar o conhecimento já adquirido pelo modelo para reduzir o tempo de treinamento e melhorar a performance, especialmente quando se tem um conjunto de dados limitado para a nova tarefa.

## Como Funciona?

1. **Modelo Pré-Treinado**: Um modelo é treinado em um conjunto de dados grande e geral, como ImageNet para reconhecimento de imagens.
   
2. **Reutilização do Modelo**: Em vez de treinar um modelo do zero, você reutiliza esse modelo pré-treinado.

3. **Adaptar para Nova Tarefa**: O modelo é ajustado para a nova tarefa, geralmente congelando as camadas iniciais (que capturam características gerais) e treinando as camadas finais para aprender os padrões específicos da nova tarefa.

## Benefícios do Transfer Learning

- **Menos Dados Necessários**: Não é necessário um grande volume de dados para treinar o modelo, pois ele já aprendeu boas representações em uma tarefa anterior.
- **Redução no Tempo de Treinamento**: O modelo já está parcialmente treinado, o que acelera o processo de treinamento.
- **Melhor Performance**: Aproveitar o conhecimento de um modelo pré-treinado pode resultar em melhor performance, especialmente em tarefas onde o treinamento do zero seria difícil ou dispendioso.

## Como Aplicar Transfer Learning?

1. **Carregar um Modelo Pré-Treinado**: Utilize um modelo já treinado em uma grande base de dados. Exemplos incluem VGG16, ResNet, Inception, etc.
   
2. **Congelar Camadas Iniciais**: As camadas do modelo que capturam características gerais (como bordas e texturas) são "congeladas", ou seja, seus pesos não são atualizados durante o treinamento na nova tarefa.

3. **Treinar as Camadas Finais**: Apenas as camadas finais (geralmente totalmente conectadas) são treinadas para a nova tarefa.

4. **Ajuste Fino (Fine-Tuning)**: Caso necessário, você pode "descongelar" algumas camadas intermediárias e ajustá-las para melhorar o desempenho.

## Exemplos de Transfer Learning

Aqui está um exemplo simples utilizando o TensorFlow:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Carregar modelo pré-treinado
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
base_model.trainable = False

# Construir o modelo final
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo com seus dados
model.fit(train_data, train_labels, epochs=5)
