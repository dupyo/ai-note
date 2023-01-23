# 블록체인 기반 딥러닝 롤백 기능

학습마다 블록을 생성하여 학습 결과에 오류가 있는 경우, 블록의 내용을 확인하여 이전 시점 중 가장 높은 정확도를 가지는 상태의 학습 결과를 가져와 해당 지점으로 복구하도록 설계 및 구현한다.

![image](https://user-images.githubusercontent.com/42468263/214069751-b7b78434-f8c8-4e4b-ab9a-3545afee10f2.png)

- 우선 학습 결과에 오류가 발생할 때 Smart contract를 사용하여 Rollback module에 작업을 요청한다. 
- Smart contract가 실행되면 각 노드들은 자신의 블록체인의 내용을 확인하고 복구할 지점을 선택하여 Rollback module에 전달한다. 
- Rollback module은 블록체인 네트워크의 모든 노드로부터 복구 지점 정보를 수집한다. 
- 수집한 정보를 검사하여 가장 높은 정확도를 가지는 지점으로 롤백을 수행한다.
