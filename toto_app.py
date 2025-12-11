import streamlit as st
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime, timedelta
from io import StringIO
from openai import OpenAI

# 페이지 설정
st.set_page_config(page_title="토토고 AI 대시보드", page_icon="🏀", layout="wide")

# ==========================================
# 🔒 보안: 비밀번호 체크 (가장 먼저 실행)
# ==========================================
def check_password():
    """비밀번호가 맞는지 확인하는 함수"""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True  # 이미 로그인 성공함

    # 비밀번호 입력창
    st.title("🔒 토토고 접근 제한")
    password = st.text_input("비밀번호를 입력하세요", type="password")
    
    if st.button("로그인"):
        # 금고(Secrets)에 저장된 비밀번호와 비교
        if password == st.secrets["app_password"]:
            st.session_state.password_correct = True
            st.rerun() # 화면 새로고침
        else:
            st.error("❌ 비밀번호가 틀렸습니다.")
    return False

if not check_password():
    st.stop() # 비밀번호 틀리면 여기서 멈춤 (아래 코드 실행 안 됨)

# ==========================================
# 🔑 API 키 불러오기 (금고에서 꺼내기)
# ==========================================
try:
    DEEPSEEK_API_KEY = st.secrets["deepseek_api_key"]
    ODDS_API_KEYS = st.secrets["odds_api_keys"]
except Exception as e:
    st.error("❌ Secrets 설정이 안 되어 있습니다. Streamlit 설정 메뉴에서 API 키를 넣어주세요.")
    st.stop()

# --- 사이드바 설정 ---
st.sidebar.title("⚙️ 설정 (Settings)")
st.sidebar.success("✅ 로그인 완료")
st.sidebar.markdown("---")
min_bet_odds = st.sidebar.slider("최소 배당", 1.1, 3.0, 1.7)
confidence_limit = st.sidebar.slider("AI 확신도", 0.5, 0.9, 0.60)
st.sidebar.markdown("---")

# --- 메인 화면 ---
st.title("🏀 토토고(TotoGo) AI 승부사")

# 함수 정의 (캐싱 최적화)
@st.cache_resource
def load_model():
    try:
        model = XGBClassifier()
        model.load_model("totogo_model.json")
        return model
    except: return None

@st.cache_data(ttl=3600)
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
        st.error("❌ 'totogo_model.json' 파일이 없습니다.")
        st.stop()
        
    # 2. 딥시크 연결
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
    # 3. 데이터 수집
    injury_db = get_injury_data()
    
    with st.spinner("🌍 전 세계 배당 정보를 수집하고 있습니다..."):
        games_data = None
        used_key = ""
        
        # 금고에서 꺼낸 키 3개를 돌려가며 사용
        for key in ODDS_API_KEYS:
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
    st.success(f"✅ 데이터 수신 완료! (보안 연결됨)")
    
    sorted_games = sorted(games_data, key=lambda x: x['commence_time'])
    limit_date = datetime.utcnow() + timedelta(hours=9, days=1)
    
    count = 0
    for game in sorted_games:
        utc_time_str = game['commence_time'].replace('Z', '')
        kst_time = datetime.fromisoformat(utc_time_str) + timedelta(hours=9)
        if kst_time.date() > limit_date.date(): continue
        
        count += 1
        home = game['home_team']
        away = game['away_team']
        
        # 데이터 추출
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
            
            # Spread
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
            
        # 카드 UI
        with st.container():
            st.markdown(f"### ⏰ {kst_time.strftime('%m/%d %H:%M')} | {home} vs {away}")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("홈팀 배당", odds_h, f"핸디 {handicap_pt_h}")
            with col2: st.metric("원정팀 배당", odds_a, "VS")
            with col3: st.metric("언오버 기준", total_pt)
            
            st.write(f"**🤖 AI 승률 예측 (홈팀 기준): {win_prob:.1f}%**")
            st.progress(int(win_prob))
            
            if color == "green": st.success(f"**{recommendation}** (배당 {odds_h})")
            elif color == "blue": st.info(f"**{recommendation}** (배당 {odds_a})")
            else: st.warning(f"**{recommendation}** - 메리트가 부족합니다.")
                
            if color != "grey":
                with st.expander("💬 딥시크 브리핑 보기"):
                    with st.spinner("작성 중..."):
                        briefing = ask_deepseek(client, {
                            'home': home, 'away': away, 'odds_h': odds_h, 'odds_a': odds_a,
                            'handicap_pt_h': handicap_pt_h, 'handicap_odds_h': handicap_odds_h,
                            'total_pt': total_pt, 'win_prob': round(win_prob, 1)
                        }, recommendation)
                        st.write(briefing)
            st.markdown("---")

    if count == 0: st.warning("📅 오늘/내일 예정된 경기가 없습니다.")
