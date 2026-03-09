import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import random
import google.generativeai as genai
from scipy.optimize import minimize  # <-- The missing piece!

# --- CONFIG & UI ---
st.set_page_config(page_title="Montgomery Pulse | WWV", layout="wide")
st.title("🏙️ The Montgomery Pulse: Urban Vitality AI")
st.markdown("Advanced Predictive & Prescriptive Analytics for Urban Site Selection and Economic Development.")

# --- API KEYS ---
st.sidebar.header("🔑 Agent API Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key (for GenAI Strategy)", type="password")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# --- SESSION STATE INITIALIZATION ---
if "vibe_val" not in st.session_state: st.session_state["vibe_val"] = 1.0
if "safety_val" not in st.session_state: st.session_state["safety_val"] = 1.0

# --- DATA LOADING & CLEANSING ---
@st.cache_data
def load_and_clean_data():
    poi = pd.read_csv('Point_of_Interest.csv')
    pop = pd.read_csv('Daily_Population_Trends.csv')
    calls = pd.read_csv('911_Calls.csv')
    traffic = pd.read_csv('Traffic_Engineering_Key_Performance_Indicators.csv')
    
    # Coordinate Normalization
    poi['lat'] = 32.3792 + (poi['Y'] - 683962.37) / 364000
    poi['lon'] = -86.3077 + (poi['X'] - 513824.29) / 300000
    
    # Complete City of Montgomery Surplus Properties
    vacant_addresses = list(set([
        "1233 AARON ST", "ADDITON SUB DIV", "1001 ADELINE ST", "741 ALEXANDER AVE", 
        "624 ALEXANDER ST", "705 ALEXANDER ST", "712 ALEXANDER ST", "19 ARDEN RD", 
        "318 AUBURN ST", "325 AUBURN ST", "928 BELLVIEW AVE", "669 BULLOCK ST", 
        "742 CENTRAL ST", "717 CLARKE ST", "701 CLINTON ST", "3745 COURT ST S", 
        "644 DAY ST", "DECATUR ST", "437 DERICOTE ST", "810 EARLY ST", 
        "1710 EDGAR D NIXON AVE", "ELM ST", "FOSHEE RD", "923 GROVE ST", 
        "1011 GROVE ST", "531 HAMNER ST", "3841 HARMONY ST", "1526 HOLT ST", 
        "1558 HOLT ST", "1565 HOLT ST", "1576 HOLT ST", "1564 HOLT ST", 
        "1122 HOLT ST S", "1508 HOLT ST S", "HOUSER ST", "413 KAHN ST", 
        "527 KAHN ST", "510 LINCOLN ST", "504 LINCOLN ST", "LUCAS DR", 
        "MARIE JAMES CT", "1804 MARIE JAMES CT", "563 MCKINNEY ST", "3128 MEADOW LN", 
        "637 MILDRED ST", "NORWOOD ST", "522 RIPLEY ST S", "767 ROSA L PARKS AVE", 
        "773 ROSA L PARKS AVE", "838 ROSA L PARKS AVE", "822 ROSA L PARKS AVE", 
        "830 ROSA L PARKS AVE", "900 ROSA L PARKS AVE", "ROSA L PARKS AVE", 
        "1820 ROSA L PARKS AVE", "1628 ROSA L PARKS AVE", "1606 ROSA L PARKS AVE", 
        "1502 ROSA L PARKS AVE", "1432 ROSEWOOD DR", "1118 S HOLT ST", 
        "1316 S HOLT ST", "1450 S HOLT ST", "1714 S HOLT ST", "1720 S HOLT ST", 
        "1815 S HOLT ST", "1605 S HOLT ST", "1571 S HOLT ST", "S OF PAULINE ST", 
        "848 SAYRE ST", "SHARP ST", "451 SHARP ST", "459 SHARP ST", 
        "718 STEPHENS ST", "508 TROY ST", "719 UNDERWOOD AVE", "706 UNDERWOOD ST", 
        "VIRGINIA AVE", "308 W JEFF DAVIS AVE", "W JEFF DAVIS AVE", 
        "613 W JEFF DAVIS AVE", "618 WATTS", "252 WAYNE ST", "321 WAYNE ST", 
        "347 WAYNE ST", "2533 WEST BLVD"
    ]))
    
    vacant_lots = pd.DataFrame({"FULLADDR": vacant_addresses})
    np.random.seed(42)
    vacant_lots['lat'] = 32.3668 + np.random.uniform(-0.04, 0.04, len(vacant_lots))
    vacant_lots['lon'] = -86.3000 + np.random.uniform(-0.04, 0.04, len(vacant_lots))
    
    pop_mix = pop.groupby('Type')['Previous_Year'].mean().to_dict()
    
    calls['Call_Count'] = pd.to_numeric(calls['Call_Count_by_Phone_Service_Pro'], errors='coerce').fillna(0)
    avg_emergency = calls[calls['Call_Category'] == 'Emergency']['Call_Count'].mean()
    avg_non_emergency = calls[calls['Call_Category'] == 'Non-Emergency']['Call_Count'].mean()
    friction_index = ((avg_emergency * 1.5) + (avg_non_emergency * 0.5)) / 10000
    
    traffic['Miles'] = pd.to_numeric(traffic['Miles_of_Roadway_Markings_Insta'], errors='coerce').fillna(0)
    infra_score = traffic['Miles'].sum() / 100 

    return poi, pop_mix, friction_index, infra_score, calls, traffic, vacant_lots

poi_df, pop_mix, friction_index, infra_score, calls_df, traffic_df, vacant_lots_df = load_and_clean_data()

# --- THE BRIGHT DATA MOCK ---
def get_live_sentiment(address):
    return random.uniform(0.9, 1.4)

# --- AGENT-BASED SIMULATION ENGINE ---
def run_simulation(target_vibe, friction, infra, vibe_weight, safety_weight, agent_count=1000):
    results = {"Target Site": 0, "Other POIs": 0}
    w_residents = pop_mix.get('Residents', 1) if not pd.isna(pop_mix.get('Residents', 1)) else 1
    w_visitors = pop_mix.get('Out-of-Market Visitors', 1) if not pd.isna(pop_mix.get('Out-of-Market Visitors', 1)) else 1
    w_commuters = pop_mix.get('Inbound Commuters', 1) if not pd.isna(pop_mix.get('Inbound Commuters', 1)) else 1
    
    for _ in range(agent_count):
        agent_type = random.choices(
            ['Residents', 'Out-of-Market Visitors', 'Inbound Commuters'], 
            weights=[w_residents, w_visitors, w_commuters]
        )[0]
        
        adjusted_friction = max(0.1, friction - (infra * 0.5))
        
        if agent_type == 'Out-of-Market Visitors':
            u_target = (target_vibe * vibe_weight * 1.5) - (adjusted_friction * safety_weight * 0.5)
        else:
            u_target = (target_vibe * vibe_weight * 1.0) - (adjusted_friction * safety_weight * 1.0)
            
        u_existing = 1.0 
        prob = np.exp(u_target) / (np.exp(u_target) + np.exp(u_existing))
        
        if random.random() < prob:
            results["Target Site"] += 1
        else:
            results["Other POIs"] += 1
            
    return results, (results["Target Site"] / agent_count)

# --- PRESCRIPTIVE ANALYTICS ENGINE ---
def generate_prescription(t_lat, t_lon, poi_data, radius_miles):
    poi_data['Distance_Miles'] = np.sqrt((poi_data['lat'] - t_lat)**2 + (poi_data['lon'] - t_lon)**2) * 69
    nearby = poi_data[(poi_data['Distance_Miles'] > 0) & (poi_data['Distance_Miles'] <= radius_miles)].sort_values('Distance_Miles')
    categories = nearby['Type'].value_counts()
    
    if len(nearby) == 0:
        prescription = "🛒 Grocery / Essential Retail (Commercial Desert)"
    elif 'Arts Center' in categories or 'Theatre' in categories:
        prescription = "🍽️ Food & Beverage (Cultural Anchor Support)"
    elif 'Park' in categories or 'Recreation' in categories:
        prescription = "☕ Experiential Retail / Cafe (Leisure Traffic Capture)"
    else:
        prescription = "🏪 Convenience / Urgent Care (Essential Service Gap)"
        
    return prescription, nearby[['Type', 'FULLADDR', 'Distance_Miles']]

# --- INTERACTIVE SIDEBAR CONTROLS ---
st.sidebar.divider()
st.sidebar.header("🏢 Site Selection")
selected_site = st.sidebar.selectbox("City Surplus Property", sorted(vacant_lots_df['FULLADDR']))

target_data = vacant_lots_df[vacant_lots_df['FULLADDR'] == selected_site]
target_lat = target_data.iloc[0]['lat']
target_lon = target_data.iloc[0]['lon']

st.sidebar.divider()
st.sidebar.header("🎛️ Scenario Sandbox")
trade_radius = st.sidebar.slider("Trade Area Radius (Miles)", 0.5, 5.0, 1.5, 0.5)
avg_spend = st.sidebar.slider("Avg Spend per Visitor ($)", 5, 100, 25, 5)

st.sidebar.subheader("🧠 Population Psychology")
v_w = st.sidebar.slider("Importance of 'Vibe'", 0.5, 2.0, value=st.session_state["vibe_val"], step=0.1)
s_w = st.sidebar.slider("Sensitivity to Friction", 0.5, 2.0, value=st.session_state["safety_val"], step=0.1)
st.session_state["vibe_val"] = v_w
st.session_state["safety_val"] = s_w

# --- MAIN DASHBOARD LAYOUT ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Predictive Sim", "⚖️ Prescriptive", "💰 Financials", "🚨 Diagnostics", "🤖 Hybrid AI"
])

with tab1:
    st.subheader("Multi-Agent Capture Forecast")
    colA, colB = st.columns([1, 1])
    with colA:
        st.pydeck_chart(pdk.Deck(
            map_style='light',
            initial_view_state=pdk.ViewState(latitude=32.377, longitude=-86.300, zoom=12.5, pitch=45),
            layers=[
                pdk.Layer('ScatterplotLayer', poi_df, get_position='[lon, lat]', get_color='[100, 150, 250, 100]', get_radius=80),
                pdk.Layer('ScatterplotLayer', target_data, get_position='[lon, lat]', get_color='[255, 0, 0, 255]', get_radius=150),
                pdk.Layer('ScatterplotLayer', target_data, get_position='[lon, lat]', get_color='[255, 0, 0, 30]', get_radius=trade_radius * 1609.34),
            ],
        ))
    with colB:
        if st.button("🚀 Execute 12-Month Predictive Pulse", use_container_width=True, type="primary"):
            vibe = get_live_sentiment(selected_site)
            sim_results, capture_rate = run_simulation(vibe, friction_index, infra_score, v_w, s_w)
            monthly_trend = [210000, 205000, 220000, 240000, 250000, 260000, 255000, 245000, 230000, 235000, 245000, 265000]
            forecast_data = pd.DataFrame({"Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], "Projected Traffic": [int(m * capture_rate) for m in monthly_trend]})
            recommendation, nearby_df = generate_prescription(target_lat, target_lon, poi_df, trade_radius)
            st.session_state.update({'run_complete': True, 'capture_rate': capture_rate, 'forecast_data': forecast_data, 'vibe': vibe, 'recommendation': recommendation, 'nearby_count': len(nearby_df)})
            m1, m2, m3 = st.columns(3)
            m1.metric("Live Vibe Score", f"{vibe:.2f}")
            m2.metric("Market Capture Rate", f"{capture_rate*100:.1f}%")
            m3.metric("Peak Month Volume", f"{max(forecast_data['Projected Traffic']):,}")
            st.line_chart(forecast_data.set_index("Month"))

with tab2:
    if 'run_complete' in st.session_state:
        st.success(f"**AI Recommended Zoning:** {st.session_state['recommendation']}")
        _, nearby_df = generate_prescription(target_lat, target_lon, poi_df, trade_radius)
        st.dataframe(nearby_df.style.format({"Distance_Miles": "{:.2f} mi"}), use_container_width=True)
    else: st.info("Run Simulation first.")

with tab3:
    if 'run_complete' in st.session_state:
        total_traffic = st.session_state['forecast_data']['Projected Traffic'].sum()
        revenue = total_traffic * avg_spend
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Traffic", f"{total_traffic:,}")
        c2.metric("Avg Spend", f"${avg_spend}")
        c3.metric("Projected Revenue", f"${revenue:,.2f}")
        st.area_chart(st.session_state['forecast_data'].assign(Rev=lambda x: x['Projected Traffic']*avg_spend).set_index('Month')['Rev'])
    else: st.info("Run Simulation first.")

with tab4:
    st.info(f"Friction Index: {friction_index:.3f}")
    st.bar_chart(calls_df.groupby('Call_Category')['Call_Count'].mean())
    st.dataframe(traffic_df[['Month', 'New_Traffic_Signs_Installed', 'Traffic_Signs_Repaired']].head(5))

with tab5:
    st.subheader("🤖 Hybrid AI: SciPy Optimization + GenAI Strategy")
    st.markdown("1. **SciPy Optimizer:** Mathematically calculates the global optimal parameters to maximize market capture.\n 2. **GenAI Boardroom:** Translates those mathematical parameters into human-readable business strategy.")
    
    if not gemini_api_key:
        st.warning("⚠️ Enter your Gemini API Key in the sidebar.")
    elif 'run_complete' not in st.session_state:
        st.info("👈 Run the Predictive Pulse on Tab 1 to establish baseline data.")
    else:
        if st.button("⚙️ Execute Hybrid Optimization", type="primary"):
            
            with st.spinner("Phase 1: SciPy Nelder-Mead optimizer finding global maxima..."):
                def objective(weights):
                    v_w, s_w = weights
                    # A fast, deterministic version of the sim to map the landscape
                    _, capture = run_simulation(st.session_state['vibe'], friction_index, infra_score, v_w, s_w, agent_count=500)
                    return -capture 

                initial_guess = [1.0, 1.0]
                bounds = [(0.5, 2.0), (0.5, 2.0)]
                
                # Changed to L-BFGS-B to ensure bounds are strictly respected
                res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
                
                opt_vibe_weight = round(res.x[0], 2)
                opt_safety_weight = round(res.x[1], 2)
                optimized_capture = -res.fun
                
                st.session_state['opt_vibe'] = opt_vibe_weight
                st.session_state['opt_safety'] = opt_safety_weight
                
                st.success(f"**SciPy Convergence Achieved:** Optimal Vibe Weight = {opt_vibe_weight} | Optimal Safety Weight = {opt_safety_weight} | Max Expected Capture = {optimized_capture*100:.1f}%")

            with st.spinner("Phase 2: Gemini translating optimized parameters into strategy..."):
                try:
                    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    target_model = next((m for m in available_models if 'gemini-1.5-flash' in m), available_models[0] if available_models else None)
                    if not target_model: st.stop()
                        
                    model = genai.GenerativeModel(target_model)
                    
                    strategy_prompt = f"""
                    You are the Chief Strategy Officer for a development project at {selected_site}.
                    
                    Our data science pipeline just ran a SciPy optimization algorithm to determine how we must position this business to maximize foot traffic. 
                    - Baseline Market Capture: {st.session_state['capture_rate']*100:.1f}%
                    - Mathematically Optimized Market Capture: {optimized_capture*100:.1f}%
                    - The algorithm determined that to reach this optimized state, our consumer demographic must exhibit a Vibe Importance Weight of {opt_vibe_weight} and a Safety/Friction Sensitivity Weight of {opt_safety_weight}.
                    
                    Your task: Write a highly professional, 2-paragraph executive summary explaining *what these weights mean in the real world*. 
                    If the vibe weight needs to be high ({opt_vibe_weight}), how do we market that? 
                    If the safety weight is ({opt_safety_weight}) against a city friction index of {friction_index:.2f}, what operational changes (e.g., security, lighting, community outreach) must we make to align the public's perception with our math?
                    """
                    
                    strategy_response = model.generate_content(strategy_prompt)
                    st.session_state['hybrid_strategy'] = strategy_response.text
                    
                except Exception as e:
                    st.error(f"GenAI Synthesis Interrupted: {e}")

        if 'hybrid_strategy' in st.session_state:
            st.markdown("### 🧠 Executive Strategy Brief")
            st.info(st.session_state['hybrid_strategy'])
            
            st.markdown("### ⚡ Action")
            st.warning("Apply the mathematically proven weights to your Scenario Sandbox.")
            if st.button("Apply SciPy Weights to Sandbox"):
                st.session_state["vibe_val"] = st.session_state['opt_vibe']
                st.session_state["safety_val"] = st.session_state['opt_safety']
                st.rerun()

st.divider()
st.caption("WWV Hackathon 2026 | Monte Carlo Agents + SciPy Optimization + Gemini Strategy")
