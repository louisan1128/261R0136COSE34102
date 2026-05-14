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

## Example 3: recovered / bm25 / keyword

- QID: `6304690-0-2`
- Failure type: `unlabeled`
- Original question: 베르나토르가 프랑스와의 동맹에도 불구하고 우호관계를 유지했던 국가는?
- Selected query: 베르나토르 프랑스와 불구하고 우호관계 유지했던 동맹에 국가
- Answer: 영국, 러시아
- Original rank: >10; selected gold rank: 1
- Metrics: Recall@10=1, MRR=1.0, Answer F1=1.0, Reward=1.9429
- Top-1 passage: 폴레옹의 요구에 부합하여 공식적으로는 프랑스와 동맹해 대륙 봉쇄에 참가하고 영국에 전쟁을 선포했으나 동시에 영국, 러시아와 비밀리에 우호 관계를 유지했다.
- Gold passage: 영국, 러시아
- Interpretation: The keyword rewrite recovered the gold passage at rank 1 by changing the retrieval surface form.

## Example 4: recovered / dense / structured

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

## Example 5: recovered / dense / keyword

- QID: `5823782-1-0`
- Failure type: `unlabeled`
- Original question: 새로 옹립된 히사미쓰마루를 기르던 사람은?
- Selected query: 히사미쓰마루 옹립된 기르던 새로 사람
- Answer: 고데라 씨(小寺氏)
- Original rank: >10; selected gold rank: 3
- Metrics: Recall@10=1, MRR=0.3333, Answer F1=1.0, Reward=1.6667
- Top-1 passage: 귀국한 지 얼마 지나지 않은 1921년 11월 25일부터 건강이 악화된 다이쇼 천황을 대신해 대리청정을 하게 됐고 따라서 칭호도 셋쇼노미야(일본어: 摂政宮 섭정궁)가 됐다. 전 내대신 히라타 도스케와 궁내대신이 된 마키노 노부아키는 대리청정을 맡은 히로히토를 고위급 궁내관 회의에 참여시켜 회의가 끝난 뒤 회의의 요점...
- Gold passage: 고데라 씨(小寺氏)
- Interpretation: The keyword rewrite recovered the gold passage at rank 3 by changing the retrieval surface form.

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

## Example 7: recovered / dense / keyword

- QID: `6269760-9-2`
- Failure type: `temporal_numeric`
- Original question: 김영삼과 함께 3선 개헌을 막기 위해 협력한 2명의 인물은?
- Selected query: 김영삼 협력한 함께 3선 개헌 막기 위해 2명
- Answer: 현석호, 한동석
- Original rank: >10; selected gold rank: 4
- Metrics: Recall@10=1, MRR=0.25, Answer F1=1.0, Reward=1.6028
- Top-1 passage: 3선 개헌 소식이 보도될 때 다시 경무대를 방문한 김영삼은 이승만 대통령에게 "박사님, 개헌하시면 안 됩니다. 국부(國父)로 남으셔야 합니다"라고 했다. 그 당시 이승만은 80세의 노인이었는데 28세였던 젊은 김영삼의 직설적인 발언을 듣고 불쾌한 나머지 손을 떨었다 한다. 그러더니 별 말없이 뒷 문으로 나가버렸다....
- Gold passage: 현석호, 한동석
- Interpretation: The keyword rewrite recovered the gold passage at rank 4 by changing the retrieval surface form.

## Example 8: recovered / dense / prompt

- QID: `6518201-1-1`
- Failure type: `abbreviation`
- Original question: 커비 슬레이드가 출연한 CBS의 프로그램은 무엇인가?
- Selected query: 커비 슬레이드가 출연한 CBS의 프로그램은 무엇인가? 정답의 근거가 되는 문서와 핵심 사실
- Answer: 온 더 로드 위드 찰스 쿠럴트
- Original rank: >10; selected gold rank: 9
- Metrics: Recall@10=1, MRR=0.1111, Answer F1=1.0, Reward=1.5556
- Top-1 passage: 엑스박스 라이브 아케이드는 마이크로소프트가 엑스박스 및 엑스박스 360 소유자에게 제공하는 온라인 서비스다. 팩맨 같은 고전 게임은 물론, 새로 나온 아케이드 게임도 제공한다. 라이브 아케이드에서는 다른 콘솔로 출시되었던 게임들도 제공하고 있는데, 플레이스테이션용으로 나왔던 《캐슬배니아: 밤의 교향곡》이 대표적이다....
- Gold passage: 온 더 로드 위드 찰스 쿠럴트
- Interpretation: The prompt rewrite recovered the gold passage at rank 9 by changing the retrieval surface form.

## Example 9: recovered / hybrid / expanded

- QID: `6304690-9-0`
- Failure type: `temporal_numeric`
- Original question: 3천명의 사상자를 내며 우세를 점한 군대는?
- Selected query: 3천명의 사상자를 내며 우세를 점한 군대는? 3천명 사상자 내며 우세 점한 군대
- Answer: 골리친의 군대
- Original rank: >10; selected gold rank: 1
- Metrics: Recall@10=1, MRR=1.0, Answer F1=1.0, Reward=2.0
- Top-1 passage: 틈이 생겼고 덕분에 돌파구를 찾지 못하고 크게 소모되어 있었던 다부의 군대는 크라스니에 도달할 수 있었다. 골리친의 군대가 반격으로 나가 우월한 포병과 기병을 내세워 공격을 되풀이해서 6천명이 정원이었던 청년 근위대는 절반이나 되는 사상자를 내고 마을을 다시 내주었다. 토르마소프의 군대가 서쪽에서 바싹 다가와 퇴로를 차단
- Gold passage: 골리친의 군대
- Interpretation: The expanded rewrite recovered the gold passage at rank 1 by changing the retrieval surface form.

## Example 10: recovered / hybrid / llm

- QID: `6471052-1-0`
- Failure type: `unlabeled`
- Original question: 첫 번째 문서조약은 언제 쓰여졌는가?
- Selected query: 문서조약 쓰여졌는 번째 관련 문서에서 '첫 번째 문서조약은 언제 쓰여졌는가?'의 정답 근거를 찾기
- Answer: 12세기
- Original rank: >10; selected gold rank: 1
- Metrics: Recall@10=1, MRR=1.0, Answer F1=1.0, Reward=1.96
- Top-1 passage: 이러한 비음화는 다른 게르만 계통 언어들에서도 나타나지만, 그렇게 길게 지속되지는 않는다. 이러한 사항들은 12세기에 쓰여진 《첫 번째 문법조약》에 나와 있기에 알 수 있는 것이며, 만일 여기 보존되어 있지 않았다면 알려지지 못했을 것이다. 《첫 번째 문법조약》에서는 글자 위에 점 하나를 찍어서 비음화를 표시하도록 하고 
- Gold passage: 12세기
- Interpretation: The llm rewrite recovered the gold passage at rank 1 by changing the retrieval surface form.

## Example 11: recovered / hybrid / structured

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

## Example 12: recovered / hybrid / expanded

- QID: `6469791-2-0`
- Failure type: `ellipsis`
- Original question: 박기혁은 어떤 상을 수상했었는가?
- Selected query: 박기혁은 어떤 상을 수상했었는가? 수상했었는 박기혁 상을
- Answer: 골든글러브
- Original rank: >10; selected gold rank: 2
- Metrics: Recall@10=1, MRR=0.5, Answer F1=1.0, Reward=1.75
- Top-1 passage:  유격수 출장 기록을 세우고도 유격수 부문 골든 글러브 수상에 실패했다. 이는 수상자인 손시헌 조차도 그가 골든글러브를 탈 줄 알아서 꽃을 들고 축하해주러 왔는데 미안하다고 말했을 정도로 이례적인 결과였다. 이 수상논란으로 인해 골든글러브를 바꿔야 한다는 논란이 계속되었다. 한편 이순철 해설위원과 이효봉 해설위원은 그가 
- Gold passage: 골든글러브
- Interpretation: The expanded rewrite recovered the gold passage at rank 2 by changing the retrieval surface form.

## Example 13: not_recovered / bm25 / prompt

- QID: `5935700-0-2`
- Failure type: `ellipsis`
- Original question: 하현우가 되고싶어하는 직업은?
- Selected query: 하현우가 되고싶어하는 직업은? 정답의 근거가 되는 문서와 핵심 사실
- Answer: 시인
- Original rank: >10; selected gold rank: >10
- Metrics: Recall@10=0, MRR=0.0, Answer F1=1.0, Reward=0.5
- Top-1 passage: 사회위해성이 있는 범죄라고 판단되는 행위에 대해서는 공안기관이 치안관리처벌법에 기초하여, 경고, 과료, 행정구류, 허가증의 취소, 외국인에 대하여는 국외추방이라고 하는 치안관리처벌을 부과한다. 치안관리처벌의 결정은 행정불복심사의 신청이나 행정소송에 의하여 다툴 수 있다. 그 외에, 일찍이 중국에는 법의 근거가 없는...
- Gold passage: 시인
- Interpretation: Even the best logged action (prompt) did not recover the gold passage in top-10; this is a useful error case.

## Example 14: not_recovered / bm25 / original

- QID: `6141330-2-1`
- Failure type: `unlabeled`
- Original question: 서양철학은 최근에 어디에서 유래된 것으로 형성 되었는가?
- Selected query: 서양철학은 최근에 어디에서 유래된 것으로 형성 되었는가?
- Answer: 유럽
- Original rank: >10; selected gold rank: >10
- Metrics: Recall@10=0, MRR=0.0, Answer F1=1.0, Reward=0.5
- Top-1 passage: 최근, 영상 디자인이 부각되고 있는 것은 우리나라를 비롯한 세계적인 추세이다. 영화나 광고도 대사와 이야기의 전개만으로는 흥행 요소가 부족하다는 것이 중론이다. 따라서 보는 시청자들에게 시각적 즐거움과 카타르시스를 느끼게 해주는 것이 최근 영상 제작의 트렌드가 되고 있다. 이러한 트렌드는 영화와 광고 뿐 아니라 박람...
- Gold passage: 유럽
- Interpretation: Even the best logged action (original) did not recover the gold passage in top-10; this is a useful error case.

## Example 15: not_recovered / bm25 / original

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

## Example 16: not_recovered / bm25 / original

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

## Example 17: not_recovered / bm25 / expanded

- QID: `6135902-0-1`
- Failure type: `abbreviation`
- Original question: AP사양은 무엇을 사용하는가
- Selected query: AP사양은 무엇을 사용하는가 ap사양 사용하는
- Answer: 삼성 엑시노스 7420
- Original rank: >10; selected gold rank: >10
- Metrics: Recall@10=0, MRR=0.0, Answer F1=0.0303, Reward=0.0152
- Top-1 passage: 검찰의 별건수사에 의해 한만호 전 한신건영 대표로부터 불법 정치자금 9억4000여만원을 받은 혐의로 기소된 한명숙은 최후진술에서 "검찰의 기소는 서울시장 출마를 막거나 낙선시킬 의도에서 이뤄진 것"이라며 "뇌물혐의 재판에서 무죄판결이 내려지기 하루 전날 정치자금법 족쇄를 채웠고 0.6% 차이로 선거에서 졌기에 결과적...
- Gold passage: 삼성 엑시노스 7420
- Interpretation: Even the best logged action (expanded) did not recover the gold passage in top-10; this is a useful error case.

## Example 18: not_recovered / bm25 / original

- QID: `6183821-0-0`
- Failure type: `ellipsis`
- Original question: 전현희의 출신 고등학교는 어디인가?
- Selected query: 전현희의 출신 고등학교는 어디인가?
- Answer: 부산 데레사여자고등학교
- Original rank: >10; selected gold rank: >10
- Metrics: Recall@10=0, MRR=0.0, Answer F1=0.0225, Reward=0.0112
- Top-1 passage: 1945년 8월 15일 곽영주가 광복 이후 동향 선배인 이정재의 도움으로 수도경찰학교를 입교 및 수료하였다는 설은 와전된것으로 보이는데 당시 이정재는 그럴만한 위치에 있지도 않았다. 이 이야기는 TV드라마 야인시대에만 나오는 이야기인데 작가가 아마 그럴듯하게 꾸며낸 것 같다. 일본군 하사관 출신의 송요찬과 일본군 준...
- Gold passage: 부산 데레사여자고등학교
- Interpretation: Even the best logged action (original) did not recover the gold passage in top-10; this is a useful error case.
