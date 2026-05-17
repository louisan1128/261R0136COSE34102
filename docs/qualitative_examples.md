# Qualitative Examples

These examples are selected from original-query hard cases. The selected strategy is the highest-reward rewrite action for the failed retriever in the logged reward table.

## Example 1: recovered / bm25 / expanded

- QID: `6533489-6-1`
- Failure type: `ellipsis`
- Original question: 담배에는 무엇이 들어있었는가?
- Selected query: 담배에는 무엇이 들어있었는가? 들어있었는 담배에
- Answer: 수은
- Original rank: >10; selected gold rank: 1
- Metrics: Recall@10=1, MRR=1.0, Answer F1=1.0, Reward=2.0
- Top-1 passage: 것을 요구한다. 코우즈키는 후지와라에게 더욱 자세한 묘사를 듣고자 담배를 피우게 해주고, 담배에 말아 놓은 수은의 기화된 연기에 중독되어 둘 모두 죽게 된다.
- Gold passage: 수은
- Interpretation: The expanded rewrite recovered the gold passage at rank 1 by changing the retrieval surface form.

## Example 2: recovered / bm25 / keyword

- QID: `6531881-1-0`
- Failure type: `temporal_numeric`
- Original question: 서일본 여객철도 223계의 외관에서는 뭐가 다시 사라졌나?
- Selected query: 223계 외관에서 사라졌나 서일본 여객철 뭐가 다시
- Answer: 비드
- Original rank: >10; selected gold rank: 1
- Metrics: Recall@10=1, MRR=1.0, Answer F1=1.0, Reward=1.9714
- Top-1 passage: 하던 방식으로부터 동일본 여객철도의 E217계등에 채용된 외판자체의 강도를 높이는 공법으로 변경하여 외관의 비드가 다시 사라졌고, 1000번대에 차량 양단부에 남아있던 수납식 창이 폐지되고 그 대신에 창이 확대되었다. 또한 장래의 개조를 간편하게 하기 위해 측면의 제1 출입문에서 앞부분과 제3출입문 뒷부분의 구조를 따로 
- Gold passage: 비드
- Interpretation: The keyword rewrite recovered the gold passage at rank 1 by changing the retrieval surface form.

## Example 3: recovered / bm25 / llm

- QID: `6482115-41-0`
- Failure type: `unlabeled`
- Original question: 김영삼 정부 시절 개칭된 국민학교의 현재 이름은?
- Selected query: 김영삼 정부 국민학교 개칭 현재 학교 명칭
- Answer: 초등학교
- Original rank: >10; selected gold rank: 1
- Metrics: Recall@10=1, MRR=1.0, Answer F1=1.0, Reward=1.9143
- Top-1 passage: 못된 버르장머리를 고쳐 주겠다고 호언하는 등 일본의 야욕에 당당한 대통령이었다. 그리고 국민학교라는 명칭을 초등학교로 개칭하고 역사바로세우기의 일환으로 쇠말뚝뽑기로 민족의 기틀을 세웠다.
- Gold passage: 초등학교
- Interpretation: The llm rewrite recovered the gold passage at rank 1 by changing the retrieval surface form.

## Example 4: recovered / bm25 / llm

- QID: `6488411-24-0`
- Failure type: `expression_mismatch`
- Original question: 당시 노태우 정부의 감시 사실을 밝힌 인물의 이름은?
- Selected query: 노태우 정부 시절 감시 사실 폭로 인물 이름
- Answer: 윤석양
- Original rank: >10; selected gold rank: 2
- Metrics: Recall@10=1, MRR=0.5, Answer F1=1.0, Reward=1.7
- Top-1 passage: 1995년 8월 2일, 그의 측근이던 총무처 장관 서석재가 전임 대통령 중 1인이 4천억 이상의 비자금과 가명계좌를 보유했다는 의혹을 제기하였고 이어 국회의원 박계동에 의해 4천억 비자금설이 폭로되었다. 전두환·노태우의 해명을 요구하여 화제가 되었다. 같은 해 7월 검찰은 '성공한 쿠데타는 처벌할 수 없다'는 논리로...
- Gold passage: 윤석양
- Interpretation: The llm rewrite recovered the gold passage at rank 2 by changing the retrieval surface form.

## Example 5: recovered / dense / structured

- QID: `6459703-16-0`
- Failure type: `ellipsis`
- Original question: 이 대통령이 시행한 사업은?
- Selected query: 핵심어: 대통령 시행한 사업 질문: 이 대통령이 시행한 사업은? 찾을 정보: 정답 근거 문서
- Answer: 4대강정비사업
- Original rank: >10; selected gold rank: 1
- Metrics: Recall@10=1, MRR=1.0, Answer F1=1.0, Reward=1.96
- Top-1 passage:  아니라고) 생각하고 있는 것 아닌가”라며 이같이 말하고 “그래서 지금 세종시에 돈 들이고 뭐 할게 아니라 4대강정비사업에 올인하려는 것”이라고 했다. 이어 이 총재는 “세종시 수정론에 찬성하는 많은 분들, 특히 지식인층에서 많이 있는데 이분들이 하나의 편견을 가지고 있다”며 “세종시는 다 노무현 말뚝이다, 그러기 때문에
- Gold passage: 4대강정비사업
- Interpretation: The structured rewrite recovered the gold passage at rank 1 by changing the retrieval surface form.

## Example 6: recovered / dense / keyword

- QID: `6559048-13-1`
- Failure type: `expression_mismatch`
- Original question: 실질적인 책임자인 국방부장관을 해임하고 군법회의에 회부해야 한다고 주장한 사람은?
- Selected query: 국방부장관 실질적인 책임자인 해임하고 군법회의 회부해야 한다고 주장한
- Answer: 민주당 박지원 원내대표
- Original rank: >10; selected gold rank: 3
- Metrics: Recall@10=1, MRR=0.3333, Answer F1=1.0, Reward=1.6222
- Top-1 passage: 이러한 논란 끝에 이번 사건에서 군의 대응실태를 조사한 감사원은 허위보고, 음주근무 등 초기 대응에서의 총체적 문제점이 드러나 합참의장 등 장성급 장교들을 포함해 총 25명에 대해 징계를 요청했으며 이중 12명은 형사처벌이 필요하다고 밝혔다. 당시 합참의장은 만취상태에서 통제실을 이탈했으며, 비상경계태세 발령을 부하...
- Gold passage: 민주당 박지원 원내대표
- Interpretation: The keyword rewrite recovered the gold passage at rank 3 by changing the retrieval surface form.

## Example 7: recovered / dense / llm

- QID: `6482115-9-1`
- Failure type: `temporal_numeric`
- Original question: 1954년 12월, 자유당을 탈당한 김영삼이 새롭게 입당한 당의 이름은?
- Selected query: 김영삼 1954년 12월 자유당 탈당 후 입당한 정당 이름
- Answer: 민주당
- Original rank: >10; selected gold rank: 3
- Metrics: Recall@10=1, MRR=0.3333, Answer F1=1.0, Reward=1.6
- Top-1 passage: 김영삼의 정치적 노선이 계승된 정당으로는 민주당 구파→ 통일민주당→ 민자당→ 신한국당이 있다. 그의 정치적 노선의 뿌리로는 자유당과 민주당 구파를 거쳐 한국민주당에 정치적 기반을 둔다. 자유당 공천으로 국회의원에 당선되었고 자유당 계열의 인사인 국무총리 장택상의 비서관으로 정계에 입문하였으나 1960년 이후 민주당으로 당
- Gold passage: 민주당
- Interpretation: The llm rewrite recovered the gold passage at rank 3 by changing the retrieval surface form.

## Example 8: recovered / dense / prompt

- QID: `6584295-40-0`
- Failure type: `unlabeled`
- Original question: 나치 독일은 쿠르스크 전투에서 어디에 패했나?
- Selected query: 나치 독일은 쿠르스크 전투에서 어디에 패했나? 정답의 근거가 되는 문서와 핵심 사실
- Answer: 소비에트 연방
- Original rank: >10; selected gold rank: 8
- Metrics: Recall@10=1, MRR=0.125, Answer F1=1.0, Reward=1.5625
- Top-1 passage: 나폴레옹이 근위대와 함께 중계점인 크라스니 마을로 들어가는 것을 방치한 후, 그가 온 길을 이미 주변부에서 대기하던 밀로라도비치의 러시아군이 단절하고 쿠투조프에게 이를 전했다. 모스크바에서 퇴각하기 시작한 원정군을 줄곧 남쪽에서 간격을 유지하며 추격해온 쿠투조프는 크라스니에 있는 나폴레옹의 근위대를 북쪽의 골리친,...
- Gold passage: 소비에트 연방
- Interpretation: The prompt rewrite recovered the gold passage at rank 8 by changing the retrieval surface form.

## Example 9: recovered / hybrid / structured

- QID: `6469791-2-1`
- Failure type: `ellipsis`
- Original question: 현대 유니콘스가 해체된 날은?
- Selected query: 핵심어: 유니콘스 해체된 현대 날은 질문: 현대 유니콘스가 해체된 날은? 찾을 정보: 정답 근거 문서
- Answer: 3월 10일
- Original rank: >10; selected gold rank: 1
- Metrics: Recall@10=1, MRR=1.0, Answer F1=1.0, Reward=1.94
- Top-1 passage: 3월 10일 현대 유니콘스가 해체되고 우리 히어로즈로 선수단이 인계된 이후 1군에서 본격적으로 두각을 드러내어 그 해 1군 116경기에 출장했다. 그는 초반에 2루수로서 활동했는데 허구연, 이효봉, 이용철 해설위원으로부터 당시 최고 2루수 수비범위를 자랑했던 '2루수 고영민' 그 이상의 재목이라는 평가를 받았다. 야구선수
- Gold passage: 3월 10일
- Interpretation: The structured rewrite recovered the gold passage at rank 1 by changing the retrieval surface form.

## Example 10: recovered / hybrid / llm

- QID: `6488411-24-0`
- Failure type: `expression_mismatch`
- Original question: 당시 노태우 정부의 감시 사실을 밝힌 인물의 이름은?
- Selected query: 노태우 정부 시절 감시 사실 폭로 인물 이름
- Answer: 윤석양
- Original rank: >10; selected gold rank: 2
- Metrics: Recall@10=1, MRR=0.5, Answer F1=1.0, Reward=1.7
- Top-1 passage: 1995년 8월 2일, 그의 측근이던 총무처 장관 서석재가 전임 대통령 중 1인이 4천억 이상의 비자금과 가명계좌를 보유했다는 의혹을 제기하였고 이어 국회의원 박계동에 의해 4천억 비자금설이 폭로되었다. 전두환·노태우의 해명을 요구하여 화제가 되었다. 같은 해 7월 검찰은 '성공한 쿠데타는 처벌할 수 없다'는 논리로...
- Gold passage: 윤석양
- Interpretation: The llm rewrite recovered the gold passage at rank 2 by changing the retrieval surface form.

## Example 11: recovered / hybrid / llm

- QID: `6511698-0-0`
- Failure type: `unlabeled`
- Original question: 투수인 구리야마 히데키가 부상 이후 전향한 포지션은?
- Selected query: 구리야마 히데키 부상 후 전향한 야구 포지션
- Answer: 야수
- Original rank: >10; selected gold rank: 2
- Metrics: Recall@10=1, MRR=0.5, Answer F1=1.0, Reward=1.6929
- Top-1 passage: 홋카이도의 구리야마 정으로부터 관광 홍보대사를 맡아달라는 제안을 받아 구리야마 정의 주민들과 친분을 쌓고 있던 것을 계기로 사재를 털어 천연잔디의 야구장과 연습장 등을 겸해서 갖춘 전용 야구장이 2002년에 완공되었다. 그 곳에서는 소년 야구 교실과 각종 대회가 열리는 등 어린이들의 꿈을 기르는 무대로 발전해가고 있...
- Gold passage: 야수
- Interpretation: The llm rewrite recovered the gold passage at rank 2 by changing the retrieval surface form.

## Example 12: recovered / hybrid / keyword

- QID: `6549968-2-0`
- Failure type: `ellipsis`
- Original question: 정도전이 관직에 나간 년도는?
- Selected query: 정도전 관직 나간 년도
- Answer: 1363년
- Original rank: >10; selected gold rank: 3
- Metrics: Recall@10=1, MRR=0.3333, Answer F1=1.0, Reward=1.6667
- Top-1 passage: 1865년(고종 2년) 9월 대비 조씨의 건의로 다시 공신 칭호를 돌려받았다. 1865년 고종은 경복궁을 중건하고 그 설계자인 정도전의 공을 인정해 그의 관작을 회복시켜 주었으며 문헌(文憲)이라는 시호를 내렸다. 그 뒤 고종은 후손들이 사는 경기 양성현(안성군 공도면, 평택시 진위면)에 사당을 건립하였다. 고종은 정...
- Gold passage: 1363년
- Interpretation: The keyword rewrite recovered the gold passage at rank 3 by changing the retrieval surface form.

## Example 13: not_recovered / bm25 / original

- QID: `6183821-0-2`
- Failure type: `ellipsis`
- Original question: 전현희의 출신 대학은 어디인가?
- Selected query: 전현희의 출신 대학은 어디인가?
- Answer: 서울대학교
- Original rank: >10; selected gold rank: >10
- Metrics: Recall@10=0, MRR=0.0, Answer F1=1.0, Reward=0.5
- Top-1 passage: 도쿄 학예 대학의 야구부에 입단하면서 투수와 내야수로서 활약을 했다. 도쿄 신대학 야구 연맹에서는 투수로서 1학년 춘계 시즌과 2학년 춘계 시즌에서의 리그 우승을 달성했지만 오른쪽 팔꿈치에 부상 당한 이후에는 투수로서의 체력적인 한계를 느껴 야수로 전향하게 되었다. 대학 시절의 통산 성적은 투수로서는 25승 8패,...
- Gold passage: 서울대학교
- Interpretation: Even the best logged action (original) did not recover the gold passage in top-10; this is a useful error case.

## Example 14: not_recovered / bm25 / original

- QID: `6269760-43-0`
- Failure type: `expression_mismatch`
- Original question: 김영삼에 대해 자신의 아들을 위해 정치적 배신을 했다고 주장한 정치인은?
- Selected query: 김영삼에 대해 자신의 아들을 위해 정치적 배신을 했다고 주장한 정치인은?
- Answer: 이민우
- Original rank: >10; selected gold rank: >10
- Metrics: Recall@10=0, MRR=0.0, Answer F1=1.0, Reward=0.5
- Top-1 passage: 그러나 1997년 12월 18일 대선에서 IMF사태에 대한 한나라당의 책임론으로 낙선했다. 김영삼은 아들 현철을 그를 차기 국회의원 이나 정치인 등으로 염두에 두었으나 신한국당의 신임 총재로 취임했던 이회창은 김영삼측의 생각을 단호하게 거절하였다. 이 일로 김영삼과 이회창간의 미묘한 감정싸움의 발단이 되어 알력으로...
- Gold passage: 이민우
- Interpretation: Even the best logged action (original) did not recover the gold passage in top-10; this is a useful error case.

## Example 15: not_recovered / bm25 / original

- QID: `6361105-7-1`
- Failure type: `temporal_numeric`
- Original question: 2005년 열린우리당 경선 최종 당선자는?
- Selected query: 2005년 열린우리당 경선 최종 당선자는?
- Answer: 정동영
- Original rank: >10; selected gold rank: >10
- Metrics: Recall@10=0, MRR=0.0, Answer F1=1.0, Reward=0.5
- Top-1 passage: 선언한다. 이 와중에 교육부총리 물망에도 올랐으나 본인의 고사로 제외되기도 한다. 하지만 당의장 경선에서 친정동영측이 지지의사를 밝힌 뒤 다시 경선에 출마하겠다는 의사를 밝혔으며, 이에 따라 박영선, 김희선 의원등 당내의 여성 의원들은 모임을 갖고 한명숙 의원으로 후보를 단일화하기로 결정했다. 4월 3일 전당대회 결과 문
- Gold passage: 정동영
- Interpretation: Even the best logged action (original) did not recover the gold passage in top-10; this is a useful error case.

## Example 16: not_recovered / bm25 / llm

- QID: `6269760-0-2`
- Failure type: `unlabeled`
- Original question: 김영삼이 대통령으로 당선된 후 출범한 정부 이름은?
- Selected query: 김영삼 대통령 당선 후 출범한 정부 명칭
- Answer: 문민 정부
- Original rank: >10; selected gold rank: >10
- Metrics: Recall@10=0, MRR=0.0, Answer F1=1.0, Reward=0.4429
- Top-1 passage: 당시 김대중의 비자금 수사를 유보한 배경에는 악화된 경제상황 및 흉흉한 민심과 기업체들의 부도사태등등 검찰 내부에서 비자금 수사를 개시한다면 호남을 중심으로 한 국민적인 저항이 발생할 가능성이 높고 제2의 광주사태를 우려하여 수사를 중지시켰던 것으로 밝혀졌다. 또 당시 검찰총장인 김태정 검찰총장의 고향이 호남출신인...
- Gold passage: 문민 정부
- Interpretation: Even the best logged action (llm) did not recover the gold passage in top-10; this is a useful error case.

## Example 17: not_recovered / dense / original

- QID: `6269760-0-2`
- Failure type: `unlabeled`
- Original question: 김영삼이 대통령으로 당선된 후 출범한 정부 이름은?
- Selected query: 김영삼이 대통령으로 당선된 후 출범한 정부 이름은?
- Answer: 문민 정부
- Original rank: >10; selected gold rank: >10
- Metrics: Recall@10=0, MRR=0.0, Answer F1=1.0, Reward=0.5
- Top-1 passage: 김영삼은 문민정부의 성격을 "1993년 광주민중항쟁을 계승한 정부"로 규정하고 광주민주화운동 등을 민주화 운동이라는 사실을 재확인하였으며 신군부 세력에게는 광주항쟁 유혈진압의 죄도 함께 물었다는 점이 있다. 또 집권 초 검은돈 거래를 차단하기 위해 1993년 8월 12일 모든 금융은 실명으로 거래하는 금융실명제를 도...
- Gold passage: 문민 정부
- Interpretation: Even the best logged action (original) did not recover the gold passage in top-10; this is a useful error case.

## Example 18: not_recovered / dense / original

- QID: `6269760-7-0`
- Failure type: `unlabeled`
- Original question: 김영삼에게 처음으로 선거 운동 협조를 요청한 정치인은?
- Selected query: 김영삼에게 처음으로 선거 운동 협조를 요청한 정치인은?
- Answer: 장택상
- Original rank: >10; selected gold rank: >10
- Metrics: Recall@10=0, MRR=0.0, Answer F1=1.0, Reward=0.5
- Top-1 passage: 1970년대의 김영삼은 국회의원 선거에 "통합야당 밀어주어 일당독재 막아내자" 라는 공약을 걸기도 했다. 1971년 5월 6일, 신민당 당수 유진산이 5·25 국회의원 총선 후보등록 마감일인 갑자기 자신의 지역구인 서울 영등포 갑구 출마를 포기하고 전국구 1번 후보를 등록하면서 진산파동이 발생했다. 5월 7일 신민당...
- Gold passage: 장택상
- Interpretation: Even the best logged action (original) did not recover the gold passage in top-10; this is a useful error case.
