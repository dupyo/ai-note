# GAN 정리


## GAN이란?


> *GAN의 주요 아이디어는 예술 작품 위조와 비슷하다.  
> 즉, 작가가 더 유명한 다른 예술가의 예술작품을 위조하는 과정과 비슷하다.*  


GAN은 다음 그림처럼 두 개의 신경망을 동시에 훈련한다. 
생성기generator G(Z)는 작품을 위조하고 판별기discriminator D(Y)는 관찰한 진짜 작품에 기반을 두고 위조한 작품이 얼마나 진짜 같은지를 판단한다. 
D(Y)는 입력으로 Y(예를 들어 하나의 이미지)를 받아 입력 변수가 얼마나 진짜 같은지를 판단하고자 투표한다. 
일반적으로 1에 가까운 값은 ‘진짜’를 나타내고 0에 가까운 값은 ‘위조’를 나타낸다. 

G(Z)는 랜덤 노이즈 Z에서 입력을 받아 G(Z)가 생성하는 모든 것이 실제라고 생각하게 D를 속이도록 훈련한다. 
판별기 D(Y)의 훈련 목적은 참 데이터 분포에서 모든 이미지의 D(Y)는 최대화하고 참 데이터 분포에서 나온 것이 아닌 모든 이미지의 D(Y)는 최소화하는 것이다. 
따라서 G와 D는 반대 게임을 한다. 따라서 적대적 훈련adversarial training이라는 이름이 생겼다. 
여기서 G와 D는 교대로 훈련하고, G와 D 각각의 목표는 그래디언트 하강을 통해 최적화된 손실 함수로 표현된다. 

생성 모델은 지속적으로 위변조 능력을 향상시키고, 판별 모델은 위변조 인식 기능을 계속 향상시킨다. 
판별기 신경망(일반적으로 표준 컨볼루션 신경망)은 입력 이미지가 실제인지 아니면 생성된 것인지를 분류하려 한다. 
중요한 새 아이디어는 생성자가 판별기를 더 자주 속일 수 있는 방법을 학습할 수 있도록 생성기와 그 판별기 모두에게 역전파해 생성기의 매개 변수를 조정하는 것이다. 
마지막으로 생성기는 실제 이미지와 구별할 수 없는 이미지를 생성하는 방법을 학습한다. 

![그림 1 랜덤 노이즈 Z를 입력 변수로 사용할 때 생성기와 판별기의 훈련 흐름](https://user-images.githubusercontent.com/42468263/186198569-bd72f732-af2b-479f-a93a-f7cc6084eb8f.png)


물론 GAN은 두 명의 경기자가 참여한 게임에서 평형equilibrium을 이루고자 노력한다. 
여기서 평형이 무슨 의미인지 먼저 이해할 필요가 있다. 시작 시점에서 두 선수 중 하나는 다른 선수보다 더 잘하길 바란다. 
그럴 경우 다른 쪽을 향상시키게 되고, 이런 방식으로 생성기와 판별기는 서로를 향상시키게 된다. 궁극적으로 어느 한쪽도 더 이상 눈에 띄게 발전하지 않는 상태에 도달하게 된다. 

손실 함수를 도식화해 두 손실(그래디언트 손실과 판별기 손실)이 언제 정점에 도달하는지 보고 이를 확인한다. 우리는 게임이 한 방향으로 너무 치우치기를 원치 않는다. 


> *위조범이 모든 경우에 판사를 속이는 방법을 즉시 배운다면 위조범은 더 이상 배울 것이 없다.*


GAN의 수렴과 다른 종류의 GAN의 안정성에 대한 세부 사항 참고 링크 : [Convergence and Stability of GAN training](https://avg.is.tuebingen.mpg.de/projects/convergence-and-stability-of-gan-training)

GAN의 생성적 응용에서는 생성기가 판별기보다 좀 더 잘 학습하기를 원한다. 

이제 GAN 학습 방법을 자세히 알아보자. 판별기와 생성기는 모두 교대로 학습한다. 학습은 두 단계로 나눌 수 있다. 

1. 여기서는 판별기 D(x)가 학습한다. 생성기 G(z)는 랜덤 노이즈 z(이는 어떤 사이즈 분포 P(z)를 따른다)에서 가짜 이미지를 생성하는 데 사용된다. 
생성기가 만든 가짜 이미지와 훈련 데이터셋의 진짜 이미지는 모두 판별기로 전달되고, 판별기는 지도학습을 통해 가짜와 진짜를 구분하려 학습한다. 
P 데이터 (x)가 훈련 데이터셋 분포라면 판별기 신경망은 해당 목적 함수를 최대화 해 입력 데이터가 진짜일 때는 1에 가깝게, 입력 데이터가 가짜일 때는 0에 가깝게 하려 한다. 
2. 다음 단계에서는 생성기 신경망이 학습한다. 생성기의 목표는 판별기 신경망이 생성된 G(z)가 진짜인 것으로 생각하게끔 속이는 것이다. 즉, D(G(z))를 1에 가깝게 하려 한다.

두 단계는 순차적으로 반복된다. 일단 훈련이 종료되면 판별기는 더 이상 실제 데이터와 가짜 데이터를 구별할 수 없고 생성기는 훈련 데이터와 매우 유사한 데이터를 작성하는 전문가가 된다.


### 텐서플로에서 GAN을 사용한 MNIST

텐서플로에서 GAN을 사용한 MNIST 신경망의 훈련을 위해 MNIST의 필기체 숫자를 사용한다. 텐서플로 케라스 데이터셋을 사용해 MNIST 데이터에 접근한다. 이 데이터는 28 × 28 크기의 필기체 숫자 훈련 이미지가 60,000개 들어 있다. 숫자의 픽셀 값은 0~55 사이이다. 각 픽셀이 [-1, 1] 범위의 값을 갖도록 입력 값을 정규화한다.

```python
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
```

단순 다층 퍼셉트론(MLP, Multi-Layered Perceptron)을 사용하고, 이미지를 784 크기의 평평한 벡터를 만들어 입력할 것이다. 따라서 훈련 데이터 모양을 수정한다. 

```python
X_train = X_train.reshape(60000, 784)
```

### 생성기(Generator)와 판별기(Discriminator) 구축

이제 생성기와 판별기를 구축해야 한다. 생성기의 목적은 노이즈 입력을 수신하고 훈련 데이터셋과 유사한 이미지를 생성하는 것이다. 노이즈 입력의 크기는 변수 randomDim으로 설정한다. 임의의 정수 값으로 초기화 하면 되는데, 대개 100으로 값을 설정한다. 

여기서는 10으로 값을 설정하였다. 이 입력은 LeakyReLU 활성화와 함께 256개의 뉴런을 가진 밀집 계층으로 공급된다. 다음으로 512개의 은닉 뉴런이 있는 또 다른 밀집 계층을 추가한 후 1,024개의 뉴런이 있는 세 번째 은닉층과 784개의 뉴런이 있는 출력 계층을 추가한다. 은닉층의 뉴런 개수를 변경하고 성능이 어떻게 변하는지 확인할 수 있다. 그러나 출력 장치의 뉴런 개수는 훈련 이미지의 픽셀 수와 일치해야 한다.

해당 생성기는 다음과 같다.

```python
generator = Sequential()
generator.add(Dense(256, imput_dim = randomDim))
generator.add(LeakyReLU(0.2))
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(Dense(784, activation='tanh'))
```

이와 비슷하게 판별기를 구축한다. 이제 판별기는 훈련 집합이나 생성기가 만든 이미지를 가져오므로 입력 크기는 784이다. 그러나 판별기의 출력은 단일 비트며, 0은 가짜 이미지(생성기가 만든 것)을 나타내고 1은 훈련 데이터 셋의 이미지(진짜 이미지)라는 것을 나타낸다.

```python
discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=784))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))
```

### GAN 구성

다음으로 생성기와 판별기를 함께 결합해 GAN을 구성한다. GAN에서는 trainable 인수를 False로 설정해 판별기 가중치를 고정시킨다.

```python
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
```

이 둘을 훈련시키는 비결은 먼저 판별기를 따로 훈련시키는 것이다. 여기서 판별기의 손실 함수로는 이진 교차 엔트로피를 사용한다. 나중에 판별기의 가중치를 동결하고 결합된 GAN을 훈련시킨다. 이때 생성기가 훈련된다. 이번에도 손실 함수는 이진 교차 엔트로피이다. 

```python
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
gan.compile(loss='binary_crossentropy', optimizer='adam')
```

### 생성기(Generator)와 판별기(Discriminator) 훈련

이제 훈련을 시작한다. 에포크마다 랜덤 노이즈 샘플을 먼저 생성기에 공급하면 생성기는 가짜 이미지를 만든다. 생성된 가짜 이미지와 실제 훈련 이미지를 특정 레이블과 함께 배치하고 이를 사용해 주어진 배치에서 먼저 판별기를 훈련시킨다.

```python
def train(epochs=1, batchSize=128):
  batchCount = int(X_train.shape[0] / batchSize)
  print ('Epochs:', epochs)
  print ('Batch size:', batchSize)
  print ('Batch per epoch:', batchCount)
  
  for e in range(1, epochs+1):
    print ('-'*15, 'Epoch %d' % e, '-'*15)
    for _ in range(batchCount):
    # 랜덤 입력 노이즈와 이미지를 얻는다.
      noise = np.random.normal(0, 1, size=[batchSize, randomDim])
      imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

      # 가짜 MNIST 이미지 생성
      generatedImages = generator.predict(noise)
      # np.shape(imageBatch), np.shape(generatedImages) 출력
      X = np.concatenate([imageBatch, generatedImages])

      # 생성된 것과 실제 이미지의 레이블
      yDis = np.zeros(2*batchSize)
      # 편파적 레이블 평활화
      yDis[:batchSize] = 0.9

      # 판별기 훈련
      discriminator.trainable = True
      dloss = discriminator.train_on_batch(X, yDis)
```

이제 동일한 for 루프에서 생성기를 훈련시킨다. 생성기가 만든 이미지가 판별기에 의해 진짜인 것으로 판별되기를 원하므로 랜덤 벡터(노이즈)를 생성기의 입력으로 사용한다. 이 방법은 가짜 이미지를 생성한 후 판별기가 이미지를 진짜 이미지인 것으로 인식하도록 GAN을 훈련시킨다(출력 1).

```python
# 생성기 훈련
noise = np.random.normal(0, 1, size=[batchSize, randomDim])
yGen = np.ones(batchSize)
discriminator.trainable = False
gloss = gan.train_on_batch(noise, yGen)
```

원한다면 생성기와 판별기의 손실과 생성된 이미지를 저장할 수 있다. 다음으로 각 에포크에 대한 손실을 저장하고 20 에포크마다 이미지를 생성한다. 

```python
# 이 에포크의 최근 배치에서의 손실을 저장
dLosses.append(dloss)
gLosses.append(gloss)

if e == 1 or e % 20 == 0:
	saveGeneratedImages(e)
```

이제 GAN 함수를 호출해 GAN을 훈련시킬 수 있다. 다음 그래프에서 GAN이 학습하는 동안 생성기와 판별기의 손실을 도식화한 것을 볼 수 있다. 

![KakaoTalk_20220823_221831631_02.jpg](https://user-images.githubusercontent.com/42468263/189040762-7ff6181b-25e4-4a59-ae0c-8e84b2813daa.png)

여기서 만든 GAN이 생성한 필기체 숫자는 다음과 같다.

![KakaoTalk_20220823_221831631_01.jpg](https://user-images.githubusercontent.com/42468263/189040914-832f019c-83c3-4034-8ced-431656d52e32.png)

![KakaoTalk_20220823_221831631.jpg](https://user-images.githubusercontent.com/42468263/189040970-f798bbdd-f3e0-4526-a9b9-af8b9f664414.png)

앞 그림을 보면 에포크가 증가함에 따라 GAN이 생성한 필기체 숫자가 점점 더 실제처럼 되는 것을 알 수 있다.

필기체 숫자의 손실과 생성된 이미지를 표시하고자 saveGeneratedImages()와 plotLoss()라는 두 가지 헬퍼 함수를 정의한다. 해당 코드는 다음과 같다.

```python
# 각 배치에서 손실 도식화
def plotLoss(epoch):
	plt.figure(figsize=(10, 8))
	plt.plot(dLosses, label='Discriminitive loss')
	plt.plot(gLosses, label='Generative loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig('images/gan_loss_epoch_%d.png' % epoch)

# 생성된 MNIST 이미지 나열
def saveGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
	noise = np.random.normal(0, 1, size=[examples, randomDim])
	generatedImages = generator.predict(noise)
	generatedImages = generatedImages.reshape(examples, 28, 28)

	plt.figure(figsize=figsize)
	for i in range(generatedImages.shape[0]):
		plt.subplot(dim[0], dim[1], i+1)
		plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')

		plt.axis('off')
	plt.tight_layout()
	plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)
```

전체 코드는 6장의 깃허브 저장소에 있는 노트북 VanillaGAN.ipynb에서 구할 수 있다. 
<!-- 다음 절에서는 최신 GAN 아키텍처를 살펴보고 텐서플로에서 구현해본다. -->
