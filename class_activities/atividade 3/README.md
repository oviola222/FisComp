# 📄 `FNCC para treinamento de funções via Tensor Flow`

Este projeto demonstra como construir, treinar e avaliar uma FNCC utilizando utilizando os pacotes **scikit-learn** e o **TensorFlow**, para treinar a aproximação de algumas funções. Inicialmente, em `atividade3_sklearn.ipynb`, foi implementado o treinamento para as funções $\sin x$, ${\sin x}/x$ e $e^{-x^2}$ utilizando o scikit-learn em sala de aula. Logo, para essa atividade 3, vamos realizar uma implementação semelhante só que no TensorFlor.

Sendo assim, este exemplo inicial visa explicar passo a passo a implementação do código para o exemplo da função função $f(x) = \sin x$, que está no arquivo `atividade3_tensorflow.ipynb`. 

---

## 1. Importação das Bibliotecas

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
```

---

## 2. Geração dos Dados de Treinamento

```python
np.random.seed(42)
num_samples = 100
data_train = np.random.uniform(0, 2 * np.pi, num_samples).reshape(-1, 1)
sin_values_train = np.sin(data_train)

noise = np.random.normal(0, 0.1, sin_values_train.shape)
sin_values_train += noise
```

- Gera 200 pontos aleatórios entre 0 a $2\pi$ com a seed 42.
- Calcula a função $f(x)=\sin x$ para cada ponto.
- Adiciona ruído (noise) de $\pm0.1$ em $f(x)$ para simular um cenário mais realista.

---

## 3. Definição do Modelo FCNN

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(5, activation='tanh'),
    tf.keras.layers.Dense(1)  # saída contínua (linear)
])
```

- Cria a rede neural sequencial com 2 camadas ocultas (5 neurônios cada) utilizando o Keras
- Keras é uma Interface de Programação de Aplicações implementada pelo TensorFlow para criar a estrutura da rede neural, aplicando os pesos, bias e funções de ativação.
- Nesse caso, a função de ativação escolhida é a `tanh`, pois o domínio é o mesmo que a da função de treinamento.
- O `input_shape` especifica o formato dos dados de entrada: no caso é 1D, uma lista de valores de $x$.
- Por fim, como essa rede neural retorna um valor de saída, um número real contínuo (a predição para a função de treinamento), essa saída deve ter apenas **1 neurônio**, sem ativação (linear).

---

## 4. Compilação e Treinamento

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mse')

model.fit(data_train, sin_values_train, epochs=500, verbose=0)
```

- Otimizador: Adam (gradiente descendente aprimorado) com taxa de aprendizado $\alpha$ de 0.01.
- Função de perda: MSE (erro quadrático médio).
- Treinamento por 500 épocas (quantidade de iterações).
- `verbose=0` não mostra a barra de progresso com informações sobre cada epoch de treinamento, incluindo a perda (loss) e o tempo.

---

## 5. Geração dos Dados de Teste

```python
num_test_samples = 100
data_test = np.linspace(0, 4 * np.pi, num_test_samples).reshape(-1, 1)
sin_values_true = np.sin(data_test)
```

- Gera 100 pontos de teste igualmente espaçados no intervalo $[0, 4\pi]$ para a rede neural prever a função além dos pontos de treinamento.

---

## 6. Previsões e Avaliação

```python
sin_values_predicted = model.predict(data_test)
mse = np.mean(np.square(sin_values_true - sin_values_predicted))
print(f"Mean Squared Error on Test Data: {mse}")
```

- Faz previsões com o modelo treinado.
- Calcula o erro quadrático médio manualmente com NumPy.

---

## 7. Visualização dos Resultados

```python
plt.figure(figsize=(6, 5))
plt.scatter(data_train, sin_values_train, label='Training Data', alpha=0.5)
plt.plot(data_test, sin_values_true, label=r'f(x)=\sin x$', color='blue')
plt.plot(data_test, sin_values_predicted, label=r'Predicted $\sin x$', color='red')
plt.xlabel(r'$x$')
plt.ylabel(r'$\sin x$')
plt.title(r'Aproximação de $\sin x$ com FCNN (TensorFlow)')
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
```

- Plota os dados de treino com ruído, a curva real e a curva predita.
- Visualmente, permite avaliar se o modelo aprendeu bem a função.

---

## ✅ Conclusão

Com apenas TensorFlow e NumPy, conseguimos treinar uma rede neural simples para aproximar uma função matemática suave. Mesmo com ruído, o modelo é capaz de aprender a estrutura subjacente da função $f(x) = \sin x$, mostrando o eficiencia das redes neurais em tarefas de regressão num cenário realista.