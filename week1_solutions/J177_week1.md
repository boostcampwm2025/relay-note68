# Q1. 🍽 부스트캠프 캠퍼를 위한 식사 메뉴 추천 프롬프트 만들기
### `해결 과정`

#### 1. 데이터 수집
일단 해당 문제를 해결하기 위해 식사 메뉴 데이터를 구해야 한다고 생각이 들었습니다.  
그래서 찾아보던 중 **배민 트렌드 리포트**라는 페이지를 발견할 수 있었습니다.  
해당 페이지에서 특정 시기에 어떤 음식이 많이 팔리는 지, 카테고리 별로 인기 음식이 무엇인지 등에 대해 분석해놓은 리포트를 읽어볼 수 있었고,  
[`배민트렌드 2025 봄·여름편`](https://ceo.baemin.com/trend/bmtrend-2025-ss)을 통해 데이터를 수집할 수 있었습니다.
<img width="1552" height="933" alt="배민 트렌드 리포트" src="https://github.com/user-attachments/assets/e76c6ad1-80c2-4883-ba94-b1b6a74731c9" />

#### 2. OpenAI ChatGPT Assistants
데이터를 수집한 뒤, 문제를 해결하기 위한 방법으로 OpenAI ChatGPT Assistants를 이용하고자 했습니다.  
해당 페이지에서 Assistant를 생성하면 Assistant 내에서 Assistant 개인의 프롬프트를 지정해 줄 수 있고,  
프롬프트를 토대로 채팅을 나눌 수 있기 때문에 선택했습니다.
<img width="1552" height="933" alt="OpenAI ChatGPT Assistants" src="https://github.com/user-attachments/assets/4c7ecd76-2934-4f20-98ac-7e9a191b053d" />
하지만 작성한 뒤, 테스트를 하려는 순간, 오류가 발생했습니다.  
확인해보니 작년에 충전해놓은 토큰이 기간이 지나서 만료되었습니다. ㅠㅠ *(토큰이 만료되는지 몰랐습니다...)*

#### 3. OpenAI ChatGPT
그래서 기존 ChatGPT를 이용하는 방식을 선택했고,  
프롬프트와 데이터 모두를 이전 채팅에서 제공한 뒤에 질문을 하는 방식으로 구현했습니다.

##### 프롬프트
```
  너는 사용자에게 원하는 음식을 추천해주는 AI야.
  사용자의 질문을 듣고, 아래 내용 데이터를 참고해서 사용자에게 3가지 이내의 음식을 추천해주면 돼.
  "category"는 해당 음식의 종류를 의미해.
  "category_rank"는 "category" 내의 순위를 의미해.
  "name"은 해당 음식의 이름을 의미해.
```

##### 데이터
```json
  {
    "category": "백반, 죽, 국수"
    "category_rank": 1
    "name": "물냉면"
  },
  {
    "category": "백반, 죽, 국수"
    "category_rank": 2
    "name": "돼지국밥"
  },
  {
    "category": "백반, 죽, 국수"
    "category_rank": 3
    "name": "순대국밥"
  },

  {
    "category": "카페, 디저트"
    "category_rank": 1
    "name": "요거트아이스크림"
  },
  {
    "category": "카페, 디저트"
    "category_rank": 2
    "name": "아이스크림"
  },
  {
    "category": "분식"
    "category_rank": 3
    "name": "인절미빙수"
  },

  {
    "category": "분식"
    "category_rank": 1
    "name": "국물떡볶이"
  },
  {
    "category": "분식"
    "category_rank": 2
    "name": "치즈떡볶이"
  },
  {
    "category": "분식"
    "category_rank": 3
    "name": "로제떡볶이"
  },

  {
    "category": "돈까스, 회, 일식"
    "category_rank": 1
    "name": "모둠초밥"
  },
  {
    "category": "돈까스, 회, 일식"
    "category_rank": 2
    "name": "모둠회"
  },
  {
    "category": "돈까스, 회, 일식"
    "category_rank": 3
    "name": "등심돈까스"
  },

  {
    "category": "치킨"
    "category_rank": 1
    "name": "후라이드치킨"
  },
  {
    "category": "치킨"
    "category_rank": 2
    "name": "양념치킨"
  },
  {
    "category": "치킨"
    "category_rank": 3
    "name": "반반치킨"
  },

  {
    "category": "야식"
    "category_rank": 1
    "name": "직화닭발"
  },
  {
    "category": "야식"
    "category_rank": 2
    "name": "국물닭발"
  },
  {
    "category": "야식"
    "category_rank": 3
    "name": "야채곱창볶음"
  },

  {
    "category": "피자"
    "category_rank": 1
    "name": "페퍼로니피자"
  },
  {
    "category": "피자"
    "category_rank": 2
    "name": "콤비네이션피자"
  },
  {
    "category": "피자"
    "category_rank": 3
    "name": "불고기피자"
  },

  {
    "category": "아시안"
    "category_rank": 1
    "name": "마라탕"
  },
  {
    "category": "아시안"
    "category_rank": 2
    "name": "소고기쌀국수"
  },
  {
    "category": "아시안"
    "category_rank": 3
    "name": "꿔바로우"
  },

  {
    "category": "양식"
    "category_rank": 1
    "name": "알리오올리오"
  },
  {
    "category": "양식"
    "category_rank": 2
    "name": "1인파스타"
  },
  {
    "category": "양식"
    "category_rank": 3
    "name": "크림파스타"
  },

  {
    "category": "족발, 보쌈"
    "category_rank": 1
    "name": "족발"
  },
  {
    "category": "족발, 보쌈"
    "category_rank": 2
    "name": "돼지보쌈"
  },
  {
    "category": "족발, 보쌈"
    "category_rank": 3
    "name": "불족발"
  },

  {
    "category": "중식"
    "category_rank": 1
    "name": "마라탕"
  },
  {
    "category": "중식"
    "category_rank": 2
    "name": "짬뽕"
  },
  {
    "category": "중식"
    "category_rank": 3
    "name": "짜장"
  },

  {
    "category": "찜, 탕, 찌개"
    "category_rank": 1
    "name": "아구찜"
  },
  {
    "category": "찜, 탕, 찌개"
    "category_rank": 2
    "name": "김치찜"
  },
  {
    "category": "찜, 탕, 찌개"
    "category_rank": 3
    "name": "삼겹살김치찜"
  },

  {
    "category": "고기, 구이"
    "category_rank": 1
    "name": "삼겹살구이"
  },
  {
    "category": "고기, 구이"
    "category_rank": 2
    "name": "고기구이"
  },
  {
    "category": "고기, 구이"
    "category_rank": 3
    "name": "제육볶음"
  },

  {
    "category": "패스트푸드"
    "category_rank": 1
    "name": "불고기버거세트"
  },
  {
    "category": "패스트푸드"
    "category_rank": 2
    "name": "단품버거"
  },
  {
    "category": "패스트푸드"
    "category_rank": 3
    "name": "치킨버거세트"
  },

  {
    "category": "도시락"
    "category_rank": 1
    "name": "치킨마요덮밥"
  },
  {
    "category": "도시락"
    "category_rank": 2
    "name": "제육볶음"
  },
  {
    "category": "도시락"
    "category_rank": 3
    "name": "항정살덮밥"
  }
```

#### 4. 결과
아래와 같은 질문을 하고, 대답을 제공받으며 해당 문제를 해결할 수 있었습니다.
##### question
```
  내가 오늘 점심을 무엇을 먹을 지 못 정했는데, 오늘 같이 더운날 어떤 음식을 먹는게 좋을 지 추천해줘.
```

##### answer
```
  더운 날 점심이라면 시원하거나 부담 없이 먹을 수 있는 음식이 제격이죠!
  아래 3가지 메뉴를 추천드릴게요:

  🥇 물냉면

  시원한 육수에 쫄깃한 면발, 무더위에 딱 맞는 대표적인 여름 음식입니다.

  더운 날 입맛 없을 때 먹기 좋아요!

  🥈 요거트아이스크림 (디저트용)

  식사 후에 가볍게 입가심으로 먹기 좋아요.

  시원하고 상큼해서 더위도 식히고 기분도 좋아집니다.

  🥉 돼지국밥

  살짝 뜨겁지만, 진하고 든든한 국물이 속을 채워줘요.

  에너지가 떨어질 땐 뜨끈한 국밥도 좋은 선택입니다.

  더 시원하게 가고 싶으면 물냉면 쪽을, 배를 든든하게 채우고 싶으면 돼지국밥 쪽을 추천드릴게요.
  혹시 고기나 다른 종류도 궁금하시면 말씀해주세요!
```
