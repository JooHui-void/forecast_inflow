# 빅콘테스트 정확도 올리기
***
### 1. window size 조정한다
- 적당하게 5? 사용중
</t>
### 2. optimizer 변경
- adam, GRU, RNN 사용
- sgd, rmsprop 사용해보기

- 현재는 adam, GRU가 성능이 좋은 것 같음
</t>
  
###3. learning rate 변경
- 일단은 0.01 언저리가 가장 적절함
- loss 그래프가 튄다면 더 줄이는 방향 필요..
  너무 줄이면 학습이 안되고 크면  **loss가 튀는 문제가 생김**
</t>

####-> loss 를 epoch가 지남에 따라 줄여가는 방향으로 생각
현재 tf.keras.callbacks.LearningRateScheduler를 이용하여 
epoch가 10이 지나면 (예시) learning rate 에 0.9를 곱해준다던가 하는 방식 사용중..

-> lstm에는 괜찮은 것 같은데 GRU에서는 안하는것만 못하는 성능이 나옴
</t>

### 4. 모델에 레이어 추가

### 5. 데이터 전처리 방식 변경
- 그대로 넣어주는 것이 더 좋을거같음 (내생각)
- 현재 지역 1 ,2 ,3 ... 구분없이 한번에 싹 넣어줬는데 다르게 처리할 방법 생각해봐도 좋음
</t>
### 6. batch size
현재 해본 바로는 충분히 크게 잡아주는 것이 좋은듯
</t>