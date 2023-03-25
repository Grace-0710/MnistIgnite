# MnistIgnite
Pytorch의 라이브러리 중 ignite 실습 - 예시모델 : MNIST


1 model 구현 : model.py

2 Custom Data Set 생성 : data_loder.py

3 학습 : train.py

4 학습 실행 : trainer.py ( ignite를 이용 )



---------------------------------------
* zero_grad : 한 루프에서 업데이트를 위해 loss.backward()를 호출하면 각 파라미터들의 .grad 값에 변화도가 저장이 된다.
이후 다음 루프에서 zero_grad()를 하지않고 역전파를 시키면 이전 루프에서 .grad에 저장된 값이 다음 루프의 업데이트에도 간섭을 해서 원하는 방향으로 학습이 안된다고 한다.
따라서 루프가 한번 돌고나서 역전파를 하기전에 반드시 zero_grad()로 .grad 값들을 0으로 초기화시킨 후 학습을 진행해야 한다.
