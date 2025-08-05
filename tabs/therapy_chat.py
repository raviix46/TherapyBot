import gradio as gr
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load BlenderBot model
model_id = "facebook/blenderbot-1B-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_id)
model = BlenderbotForConditionalGeneration.from_pretrained(model_id)

chat_history = []

# Response function
def respond(user_message):
    global chat_history
    inputs = tokenizer([user_message], return_tensors="pt").to(model.device)
    with torch.no_grad():
        reply_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    chat_history.append((user_message, reply))
    formatted_chat = "\n\n".join([f"👤 You: {u}\n\n🤖 Bot: {b}\n\n------" for u, b in chat_history])
    return "", formatted_chat.strip()

# Clear function
def clear_chat():
    global chat_history
    chat_history = []
    return "", ""

# UI function
# UI function
def therapy_chat_tab():
    gr.Markdown("## 🧠 TherapyBot++", elem_classes="centered-text")
    gr.Markdown(
        "Thoughtfully designed to be your safe and supportive space — here for you when your mind needs it the most.",
        elem_classes="centered-text"
    )

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(placeholder="How are you feeling?", label="🗣 Your Message", lines=8)
            submit_btn = gr.Button("Submit", elem_id="submit-btn")
            clear_btn = gr.Button("Clear")
        with gr.Column():
            with gr.Accordion("📜 Chat History", open=True):
                chat_display = gr.Textbox(label=None, interactive=False, lines=13.8, container=False)

    submit_btn.click(respond, user_input, outputs=[user_input, chat_display])
    clear_btn.click(clear_chat, outputs=[user_input, chat_display])

    gr.Markdown(
        """
        <div style="
            height: auto;
            min-height: 100px;
            width: 100%;
            max-width: 65%;
            margin: 15px auto;
            overflow: hidden;
            position: relative;
            background: linear-gradient(to right, #1c2f3a 0%, #000000 50%, #1c2f3a 100%);
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.4);
            text-align: center;
            padding: 16px 24px;
            box-sizing: border-box;
        ">            
            <div style="
                font-size: 15px; 
                color: #e0e0e0; 
                font-weight: 600; 
                margin-bottom: 14px; 
                position: relative; 
                z-index: 2;
            ">
                🧠 <em>Here are some surprising and helpful wellness facts:</em>
            </div>
            <div class="scroll-wrapper chatbot-scroll">
                <div class="scroll-content chatbot-content">
                    <div class="scroll-inner">
                        <div>-----</div>
                        <div>🧠 Talking to someone about your emotions can significantly lower stress levels.</div>
                        <div>👃 The average human nose can identify over 50,000 different smells.</div>
                        <div>😶‍🌫️ Suppressing emotions doesn’t erase them — they return stronger later.</div>
                        <div>🧠 Your brain remains plastic at any age — emotional healing is always possible.</div>
                        <div>🍋 Lemons are among the healthiest fruits, rich in antioxidants and immune support.</div>
                        <div>🛏️ Sleep affects nearly every system in the body — it’s your natural mental reset.</div>
                        <div>🔁 Replaying negative thoughts rewires your brain for anxiety — called cognitive looping.</div>
                        <div>🍵 Drinking a warm beverage may help your body feel cooler and more relaxed.</div>
                        <div>👶 Newborns have the fastest heartbeats of any human age group.</div>
                        <div>🦶 Your feet can reveal early signs of deeper health issues like diabetes or stress.</div>
                        <div>👫 Companionship and bonding are great for both your heart and brain health.</div>
                        <div>👂 The brain releases calming chemicals when it feels truly listened to.</div>
                        <div>🌞 Optimistic thinking can add years to your life by reducing stress.</div>
                        <div>❄️ Cold weather can boost metabolism and strengthen your immune system.</div>
                        <div>👁️ You can physically see signs of high cholesterol in your eyes and skin.</div>
                        <div>🎨 Creative activities like drawing or music reduce anxiety and boost cognition.</div>
                        <div>🧬 Trauma can be inherited — emotions may echo generations before you.</div>
                        <div>😢 Humans are the only animals who cry due to emotional reasons.</div>
                        <div>📵 Limiting screen time improves focus, clarity, and emotional regulation.</div>
                        <div>💓 Your heart beats over 100,000 times a day to keep you alive and well.</div>
                        <div>☕ Drinking coffee may reduce symptoms of depression and lift your mood.</div>
                        <div>💧 You can lose 3% body weight in fluids before even feeling thirsty.</div>
                        <div>📆 Heart attacks are most frequent on Monday mornings due to work stress.</div>
                        <div>🐶 A dog’s companionship may lower blood pressure and protect your heart.</div>
                        <div>🩸 The heart pumps nearly 2,000 gallons of blood every day.</div>
                        <div>😣 Chronic stress increases your risk of type 2 diabetes.</div>
                        <div>🩻 Some capillaries are ten times thinner than a strand of hair.</div>
                        <div>🦴 More than half your bones are located in your hands and feet.</div>
                        <div>🧘 Just 10 minutes of mindful breathing can reset your brain chemistry.</div>
                        <div>❓ Most therapy breakthroughs come from better questions — not quick answers.</div>
                        <div>😂 Laughter releases feel-good hormones and even helps reduce pain.</div>
                        <div>🫀 Your heart weighs around 7 to 15 ounces depending on your body.</div>
                        <div>🏃 The heart works harder than a sprinter's leg muscles — nonstop effort.</div>
                        <div>🧠 Emotional pain activates the same brain region as physical pain.</div>
                        <div>🍌 Bananas can naturally boost your mood by increasing serotonin levels.</div>
                    </div>
                </div>
            </div>
        </div>
        <style>
            .chatbot-scroll .chatbot-content {
                position: absolute;
                width: 100%;
                animation: chatbot-scroll-vertical 130s linear infinite;
                will-change: transform;
            }
            .scroll-wrapper {
                height: 50px;
                overflow: hidden;
                position: relative;
            }
            .scroll-inner {
                display: flex;
                flex-direction: column;
                gap: 20px;
                color: #bbbbbb;
                font-size: 13.8px;
                line-height: 1.5;
            }
            @keyframes chatbot-scroll-vertical {
                0% {
                    transform: translateY(0);
                }
                100% {
                    transform: translateY(-50%);
                }
            }
        </style>
        """,
        elem_classes="centered-text"
    )