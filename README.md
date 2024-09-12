Github Convention
===
main branch는 배포이력을 관리하기 위해 사용, 

BTC branch는 기능 개발을 위한 branch들을 병합(merge)하기 위해 사용

모든 기능이 추가되고 버그가 수정되어 배포 가능한 안정적인 상태라면 BTC branch에 병합(merge)

작업을 할 때에는 개인의 branch를 통해 작업

데이터 전처리팀의 branch명의 형식은 “data/기능이름” 으로 작성
ex) data/rolling, data/sort

모델팀의 branch명의 형식은 “model/기능이름”으로 작성 
ex) model/LSTM, model/LightGBM

master(main) Branch에 Pull request를 하는 것이 아닌, model Branch 또는 data Branch에 Pull request 요청

commit을 할 경우 어떤 부분을 수정하였는지 작성
ex) git commit -m “Added 기능이름 to data”
ex) git commit -m “fixed 기능이름 to data”
ex) git commit -m “deleted 기능이름 to data”

pull request merge 담당자 : data - 다빈 / model - 영균
나머지는 BTC branch 건드리지 말 것!

Pull request를 요청한 이후에 Github issue에 어떤 작업을 했는지 작성하기!
