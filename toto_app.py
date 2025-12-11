import streamlit as st
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime, timedelta
from io import StringIO
from openai import OpenAI

# ==========================================
# 🔑 [필수] API 키 설정 (앱에서 바로 수정 가능하도록 설정)
# ==========================================
# 여기에 미리 적어두셔도 되고, 실행 후 웹화면에서 입력해도 됩니다.
DEFAULT_DEEPSEEK_KEY = "sk-77093904b26643038a270043ea59cc3b"
DEFAULT_ODDS_KEYS = [
    "e5e2ea14754efa0034022ed74db1d57d",
    "9eeb85750b20d56d69544205710d6126",
    "5741cff533daa57d8dd5ab91e1ec4fe8"
]
# ==========================================

# 페이지 기본 설정 (제목, 아이콘)
st.set_page_config(page_title="토토고 AI 대시보드", page_icon="🏀", layout="wide")

# --- 사이드바: 설정 메뉴 ---
st.sidebar.title("⚙️ 설정 (Settings)")

# API 키 입력받기 (코드에 적은거 있으면 그거 쓰고, 아니면 입력창 뜸)
deepseek_key = st.sidebar.text_input("DeepSeek API Key", value=DEFAULT_DEEPSEEK_KEY, type="password")
odds_keys_input = st.sidebar.text_area("Odds API Keys (한 줄에 하나씩)", value="\n".join(DEFAULT_ODDS_KEYS))
odds_keys = [k.strip() for k in odds_keys_input.split('\n') if k.strip()]

st.sidebar.markdown("---")
min_bet_odds = st.sidebar.slider("최소 배당 (Min Odds)", 1.1, 3.0, 1.7)
confidence_limit = st.sidebar.slider("AI 확신도 (Confidence)", 0.5, 0.9, 0.60)
st.sidebar.markdown("---")
st.sidebar.info("버튼을 누르면 분석을 시작합니다.")

# --- 메인 화면 ---
st.title("🏀 토토고(TotoGo) AI 승부사")
st.markdown("### 내 손안의 AI 스포츠 베팅 에이전트")

# 함수 정의 (캐싱을 써서 속도 최적화)
@st.cache_resource
def load_model():
    try:
        model = XGBClassifier()
        model.load_model("totogo_model.json")
        return model
    except:
        return None

@st.cache_data(ttl=3600) # 1시간마다 갱신
def get_injury_data():
    url = "https://www.cbssports.com/nba/injuries/"
    header = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=header)
        dfs = pd.read_html(StringIO(r.text))
        injury_dict = {}
        for df in dfs:
            cols = df.columns.tolist()
            player_col = next((c for c in cols if 'Player' in str(c)), None)
            status_col = next((c for c in cols if 'Status' in str(c)), None)
            detail_col = next((c for c in cols if 'Injury' in str(c) and 'Status' not in str(c)), None)
            if player_col:
                for _, row in df.iterrows():
                    injury_dict[str(row[player_col])] = f"{str(row[status_col])} ({str(row[detail_col])})"
        return injury_dict
    except: return {}

def ask_deepseek(client, match_info, prediction):
    if not client: return "API 키가 없어서 브리핑을 생략합니다."
    
    prompt = f"""
    당신은 스포츠 분석가 '토토고'입니다.
    [경기] {match_info['home']} vs {match_info['away']}
    [데이터]
    - 배당: {match_info['odds_h']} vs {match_info['odds_a']}
    - 핸디캡: {match_info['handicap_pt_h']} (배당 {match_info['handicap_odds_h']})
    - 언오버: {match_info['total_pt']}
    [AI 판단] 승률 {match_info['win_prob']}% / 추천: {prediction}
    
    위 정보를 바탕으로 베터에게 줄 3줄 요약 조언을 작성하세요.
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", messages=[{"role": "user", "content": prompt}], stream=False
        )
        return response.choices[0].message.content
    except Exception as e: return f"브리핑 실패: {e}"

# --- 실행 버튼 ---
if st.button("🚀 분석 시작 (Analyze Now)", type="primary"):
    
    # 1. 모델 로드
    model = load_model()
    if not model:
        st.error("❌ 'totogo_model.json' 파일이 없습니다. 학습(toto_train.py)을 먼저 해주세요!")
        st.stop()
        
    # 2. 딥시크 클라이언트 연결
    client = None
    if deepseek_key and "여기에" not in deepseek_key:
        client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
        
    # 3. 데이터 수집
    injury_db = get_injury_data()
    
    with st.spinner("🌍 전 세계 배당 정보를 수집하고 있습니다..."):
        games_data = None
        used_key = ""
        
        for key in odds_keys:
            if "여기에" in key: continue
            url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={key}&regions=eu&markets=h2h,spreads,totals&oddsFormat=decimal'
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    games_data = r.json()
                    used_key = key[:5] + "***"
                    break
            except: pass
            
        if not games_data:
            st.error("❌ 모든 API 키가 막혔거나 오류가 발생했습니다.")
            st.stop()
            
    # 4. 분석 및 화면 표시
    st.success(f"✅ 데이터 수신 완료! (사용된 키: {used_key})")
    
    sorted_games = sorted(games_data, key=lambda x: x['commence_time'])
    limit_date = datetime.utcnow() + timedelta(hours=9, days=1) # 내일까지
    
    count = 0
    for game in sorted_games:
        # 날짜 필터
        utc_time_str = game['commence_time'].replace('Z', '')
        kst_time = datetime.fromisoformat(utc_time_str) + timedelta(hours=9)
        if kst_time.date() > limit_date.date(): continue
        
        count += 1
        home = game['home_team']
        away = game['away_team']
        
        # 데이터 추출 (간략화)
        odds_h, odds_a = 0, 0
        handicap_pt_h, handicap_odds_h = 0, 0
        total_pt = 0
        
        try:
            bookmakers = game['bookmakers']
            if not bookmakers: continue
            
            # H2H
            h2h = next((m for b in bookmakers for m in b['markets'] if m['key'] == 'h2h'), None)
            if h2h:
                odds_h = next(o['price'] for o in h2h['outcomes'] if o['name'] == home)
                odds_a = next(o['price'] for o in h2h['outcomes'] if o['name'] == away)
            
            # Spread (Home)
            spread = next((m for b in bookmakers for m in b['markets'] if m['key'] == 'spreads'), None)
            if spread:
                s_out = next((o for o in spread['outcomes'] if o['name'] == home), None)
                if s_out: handicap_pt_h, handicap_odds_h = s_out['point'], s_out['price']
                
            # Total
            total = next((m for b in bookmakers for m in b['markets'] if m['key'] == 'totals'), None)
            if total:
                t_out = next((o for o in total['outcomes'] if o['name'] == 'Over'), None)
                if t_out: total_pt = t_out['point']
        except: continue
        
        if odds_h == 0: continue

        # AI 예측
        features = pd.DataFrame({
            'odds_win': [float(odds_h)],
            'odds_lose': [float(odds_a)],
            'reference_point': [float(handicap_pt_h)] 
        })
        win_prob = model.predict_proba(features)[0][1] * 100
        
        # 추천 로직
        recommendation = "관망 (Pass)"
        color = "grey"
        if win_prob >= confidence_limit*100 and odds_h >= min_bet_odds:
            recommendation = f"🔥 홈팀({home}) 승리 추천"
            color = "green"
        elif win_prob <= (1-confidence_limit)*100 and odds_a >= min_bet_odds:
            recommendation = f"🌊 원정팀({away}) 승리 추천"
            color = "blue"
            
        # --- UI 카드 그리기 ---
        with st.container():
            st.markdown(f"### ⏰ {kst_time.strftime('%m/%d %H:%M')} | {home} vs {away}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("홈팀 배당", odds_h, f"핸디 {handicap_pt_h}")
            with col2:
                st.metric("원정팀 배당", odds_a, "VS")
            with col3:
                st.metric("언오버 기준", total_pt)
            
            # AI 결과 바
            st.write(f"**🤖 AI 승률 예측 (홈팀 기준): {win_prob:.1f}%**")
            st.progress(int(win_prob))
            
            # 추천 박스
            if color == "green":
                st.success(f"**{recommendation}** (배당 {odds_h})")
            elif color == "blue":
                st.info(f"**{recommendation}** (배당 {odds_a})")
            else:
                st.warning(f"**{recommendation}** - 메리트가 부족합니다.")
                
            # 딥시크 브리핑 (추천 경기에만 열어보기)
            if color != "grey":
                with st.expander("💬 딥시크(DeepSeek) 상세 브리핑 보기"):
                    with st.spinner("리포트 작성 중..."):
                        briefing = ask_deepseek(client, {
                            'home': home, 'away': away, 'odds_h': odds_h, 'odds_a': odds_a,
                            'handicap_pt_h': handicap_pt_h, 'handicap_odds_h': handicap_odds_h,
                            'total_pt': total_pt, 'win_prob': round(win_prob, 1)
                        }, recommendation)
                        st.write(briefing)
            
            st.markdown("---")

    if count == 0:
        st.warning("📅 오늘/내일 예정된 경기가 없습니다.")