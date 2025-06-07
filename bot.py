import logging
import requests
import pandas as pd
import datetime
import asyncio
import aiohttp
import pytz
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
import ta
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# --- Google Cloud Translation Setup ---
# Import the client library
try:
    from google.cloud import translate_v2 as translate
    # Instantiates a client
    google_translate_client = translate.Client()
    logger = logging.getLogger(__name__) # Ensure logger is defined before use
    logger.info("Google Cloud Translation API client initialized.")
    OFFICIAL_TRANSLATOR_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__) # Ensure logger is defined before use
    logger.warning("google-cloud-translate library not found. Install with: pip install google-cloud-translate")
    google_translate_client = None
    OFFICIAL_TRANSLATOR_AVAILABLE = False
except Exception as e:
    logger = logging.getLogger(__name__) # Ensure logger is defined before use
    logger.error(f"Error initializing Google Cloud Translation client: {e}. Ensure GOOGLE_APPLICATION_CREDENTIALS is set and API is enabled.")
    google_translate_client = None
    OFFICIAL_TRANSLATOR_AVAILABLE = False


console = Console()

# --- Logging Setup (Moved to top - already done in previous fix, but good to re-emphasize) ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# The logger is already defined above the try-except for the translation client.

# --- Configuration ---
BOT_TOKEN = "8071210437:AAG7vxCiR5aQAt1mGGUQj10UURqaHnIxWi0"
SYMBOL = "BTCUSDT"

# Your CryptoPanic API Token - IMPORTANT: Keep this secure!
CRYPTOPANIC_API_TOKEN = "062951aa1cbb5e40ed898a90a48b6eb1bcf3f8a7"

# Telegram File IDs for cool animations/stickers
LOADING_STICKER_ID = "CAACAgIAAxkBAAIDyma29XQyFz4Q9IqQ-aY8X6G8s4C1AAJbDwACz_OBAAFBw6b9o4R2czQE" # Example: A spinning gear sticker
SUCCESS_STICKER_ID = "CAACAgIAAxkBAAID0Ga29fJ4S-vI7_X9M-jKx5I9p8U1AAJmDwACz_OBAAFBw6b9o4R2czQE" # Example: A celebratory sticker
ERROR_STICKER_ID = "CAACAgIAAxkBAAID0ma29k7_Qj1L9O5_X-iY0yH9m8Y1AAJnDwACz_OBAAFBw6b9o4R2czQE" # Example: A sad face sticker

# News sentiment keywords
BULLISH_WORDS = ["etf", "approval", "adoption", "bullish", "rally", "growth", "positive", "uptrend", "gain", "boom", "breakout", "surge", "innovate", "partnership", "fund", "invest"]
BEARISH_WORDS = ["hack", "ban", "dump", "bearish", "lawsuit", "downfall", "negative", "downtrend", "loss", "crash", "correction", "fraud", "exploit", "regulation", "fud", "scam", "bear"]

# --- Technical Functions ---
def fetch_binance_ohlcv(symbol, interval, limit=100):
    """Fetches candlestick data from Binance."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def detect_engulfing(df):
    """Detects bullish or bearish engulfing candlestick patterns."""
    if len(df) < 2:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]

    if (last['close'] > last['open'] and 
        prev['close'] < prev['open'] and 
        last['close'] > prev['open'] and 
        last['open'] < prev['close']):
        return 'bullish'
    
    if (last['close'] < last['open'] and 
        prev['close'] > prev['open'] and 
        last['open'] > prev['close'] and 
        last['close'] < prev['open']):
        return 'bearish'
    return None

def analyze_sentiment(news_titles):
    """Analyzes the sentiment of news headlines based on keywords."""
    score = 0
    for title in news_titles:
        title = title.lower()
        for word in BULLISH_WORDS:
            if word in title:
                score += 1
        for word in BEARISH_WORDS:
            if word in title:
                if word in ["hack", "ban", "scam", "fraud", "exploit", "regulation", "fud", "crash"]: 
                    score -= 2 
                else:
                    score -= 1
    return score

async def fetch_news_sentiment():
    """Fetches news headlines from CryptoPanic and calculates sentiment, with official English translation."""
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_TOKEN}&currencies=BTC&public=true"
    processed_headlines = []
    translation_status_message = ""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if 'results' in data:
                    raw_headlines = [post['title'] for post in data['results'][:10]] # Get top 10 headlines

                    if OFFICIAL_TRANSLATOR_AVAILABLE and google_translate_client:
                        for hl in raw_headlines:
                            try:
                                result = google_translate_client.translate(hl, target_language='en')
                                translated_hl = result['translatedText']
                                processed_headlines.append(translated_hl)
                            except Exception as trans_e:
                                logger.error(f"Official translation failed for headline '{hl}': {trans_e}. Using original.")
                                processed_headlines.append(hl + " (Translation failed ❌)") # Use a more definitive error emoji
                                translation_status_message = "\n\n_❌ Some news headlines could not be translated by Google Cloud. Check API status and limits._"
                    else:
                        processed_headlines = raw_headlines
                        translation_status_message = "\n\n_ℹ️ Official news translation service not configured or available. Headlines are not guaranteed English._"
                    
                    return analyze_sentiment(processed_headlines), processed_headlines, translation_status_message
                else:
                    logger.warning("CryptoPanic API response did not contain 'results'.")
                    return 0, [], "_🚫 No news found._"
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching news from CryptoPanic API: {e}")
        return 0, [], f"_❌ Error fetching news: {e}_"
    except Exception as e:
        logger.error(f"An unexpected error occurred in fetch_news_sentiment: {e}")
        return 0, [], f"_❌ An unexpected error occurred while fetching news: {e}_"

def apply_indicators(df):
    """Applies various technical indicators to the DataFrame."""
    df['ema20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'], df['macd_signal'] = macd.macd(), macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['supertrend'] = ta.trend.STCIndicator(df['close']).stc()
    return df

def calculate_signal(df5, df1h, df4h, news_score, headlines):
    """Calculates a trading signal based on multiple indicators and news sentiment."""
    last5 = df5.iloc[-1]
    signal, reasons, score = "WAIT", [], 0

    # 4-hour trend
    if df4h['ema20'].iloc[-1] > df4h['ema50'].iloc[-1]:
        score += 1
        reasons.append("📈 *4H Trend:* Bullish (EMA20 > EMA50)")
    else:
        score -= 1
        reasons.append("📉 *4H Trend:* Bearish (EMA20 < EMA50)")

    # 1-hour trend
    if df1h['ema20'].iloc[-1] > df1h['ema50'].iloc[-1]:
        score += 1
        reasons.append("📊 *1H Trend:* Bullish (EMA20 > EMA50)")
    else:
        score -= 1
        reasons.append("📊 *1H Trend:* Bearish (EMA20 < EMA50)")

    # 5-minute RSI
    if last5['rsi'] < 35:
        score += 1
        reasons.append("✅ *RSI:* Oversold (Potential Buy)")
    elif last5['rsi'] > 65:
        score -= 1
        reasons.append("🛑 *RSI:* Overbought (Potential Sell)")
    else:
        reasons.append("↔️ *RSI:* Neutral (35-65)")

    # 5-minute MACD
    if last5['macd'] > last5['macd_signal']:
        score += 1
        reasons.append("⬆️ *MACD:* Bullish Crossover")
    else:
        score -= 1
        reasons.append("⬇️ *MACD:* Bearish Crossover")

    # 5-minute Engulfing Pattern
    engulf = detect_engulfing(df5)
    if engulf == 'bullish':
        score += 1
        reasons.append("🔥 *Candlestick:* Bullish Engulfing Pattern!")
    elif engulf == 'bearish':
        score -= 1
        reasons.append("🥶 *Candlestick:* Bearish Engulfing Pattern!")
    else:
        reasons.append("🕯️ *Candlestick:* No Engulfing Pattern Detected")

    # 5-minute Supertrend (based on the `stc` indicator's value from `ta` library)
    if not pd.isna(last5['supertrend']):
        if last5['supertrend'] > last5['close']:
            score -= 1
            reasons.append("📉 *Supertrend (STC):* Bearish Signal")
        else:
            score += 1
            reasons.append("📈 *Supertrend (STC):* Bullish Signal")
    else:
        reasons.append("❓ *Supertrend (STC):* Data not available")

    # News Sentiment
    if news_score > 0:
        score += 1
        reasons.append(f"📰 *News Sentiment:* Positive (Score: {news_score})")
    elif news_score < 0:
        score -= 1
        reasons.append(f"📰 *News Sentiment:* Negative (Score: {news_score})")
    else:
        reasons.append("📰 *News Sentiment:* Neutral")

    # Determine final signal based on aggregate score
    if score >= 3:
        signal = "BUY"
    elif score <= -3:
        signal = "SELL"
    else:
        signal = "WAIT"

    # Calculate Stop Loss (SL) and Take Profit (TP) using ATR
    sl, tp = None, None
    if not pd.isna(last5['atr']) and last5['atr'] > 0:
        if signal == "BUY":
            sl = last5['close'] - (2 * last5['atr'])
            tp = last5['close'] + (4 * last5['atr'])
        elif signal == "SELL":
            sl = last5['close'] + (2 * last5['atr'])
            tp = last5['close'] - (4 * last5['atr'])

    return signal, reasons, last5['close'], sl, tp, score, headlines

async def fetch_and_decide():
    """Fetches all necessary data, applies indicators, and generates a trading decision."""
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[bold magenta]🚀 Analyzing BTC/USDT Market...[/bold magenta]", total=100)
        
        df5 = apply_indicators(fetch_binance_ohlcv(SYMBOL, "5m", limit=100))
        progress.update(task, advance=25, description="[green]📊 Fetching 5m data...[/green]")
        
        df1h = apply_indicators(fetch_binance_ohlcv(SYMBOL, "1h", limit=100))
        progress.update(task, advance=25, description="[yellow]📈 Fetching 1h data...[/yellow]")
        
        df4h = apply_indicators(fetch_binance_ohlcv(SYMBOL, "4h", limit=100))
        progress.update(task, advance=25, description="[cyan]📉 Fetching 4h data...[/cyan]")
        
        # Call fetch_news_sentiment to get news and its translation status
        news_score, headlines, translation_status_message = await fetch_news_sentiment()
        progress.update(task, advance=25, description="[blue]📰 Analyzing news sentiment & translating...[/blue]")
        
        # Pass both the signal calculation results AND the translation status message
        return calculate_signal(df5, df1h, df4h, news_score, headlines), translation_status_message

# --- Telegram Bot Handlers ---
async def start(update: Update, context):
    """Handles the /start command."""
    welcome_message = (
        "👋 *Welcome to the Lucky Sir BTC/USDT Bot!* ✨\n\n"
        "I'm your personal crypto analyst, providing *real-time trading signals* "
        "for *BTC/USDT* by combining deep market analysis with instant news sentiment.\n\n"
        "Are you ready to unlock potential trading opportunities? Tap below! 👇"
    )
    await update.message.reply_text(
        welcome_message,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("⚡️ GET INSTANT SIGNAL! 🚀", callback_data="signal")]
        ])
    )

async def callback_handler(update: Update, context):
    """Handles callback queries from inline keyboard buttons."""
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id

    animated_messages = [
        ("🔄 *Initiating Market Scan...* [⏳]", 0.8),
        ("✨ *Collecting Latest Candlesticks...* [📊]", 0.8),
        ("🌐 *Scanning Global News Feeds...* [📰]", 0.8),
        ("🧠 *Running Advanced AI Analysis...* [💡]", 0.8),
        ("🚀 *Generating Your Signal...* [⚙️]", 0.8)
    ]

    for msg, delay in animated_messages:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await query.edit_message_text(msg, parse_mode='Markdown')
        await asyncio.sleep(delay)

    if LOADING_STICKER_ID:
        try:
            sent_sticker = await context.bot.send_sticker(chat_id=chat_id, sticker=LOADING_STICKER_ID)
            await asyncio.sleep(1.5)
        except Exception as e:
            logger.error(f"Failed to send loading sticker: {e}")

    # Unpack both the signal details and the translation status message
    (signal, reasons, price, sl, tp, score, headlines), translation_status_message = await fetch_and_decide()
    
    ist_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')

    signal_highlight = {
        "BUY": "🟢 *STRONG BUY!* 📈",
        "SELL": "🔴 *STRONG SELL!* 📉",
        "WAIT": "⚪ *WAIT / NEUTRAL* ↔️"
    }.get(signal, signal)

    sl_tp_text = ""
    if signal in ["BUY", "SELL"] and sl is not None and tp is not None:
        sl_tp_text = (
            f"\n\n🚨 *Risk Management:*\n"
            f"   📉 *Stop Loss (SL):* `{sl:.2f}`\n"
            f"   🎯 *Take Profit (TP):* `{tp:.2f}`"
        )
    
    reasons_text = "\n".join(reasons)
    if not reasons_text:
        reasons_text = "_No specific technical or news reasons found at this moment._"

    news_list_text = "\n".join([f"• {h}" for h in headlines[:5]])
    if not news_list_text:
        news_list_text = "🚫 _No significant news found recently._"
    
    # Construct the news section with the translation status message appended
    news_section = (
        f"📰 *Top News Highlights:*\n"
        f"{news_list_text}"
        f"{translation_status_message}" # Appends the status message (e.g., if translation failed)
    )

    final_message = (
        f"🌟 *BTC/USDT Market Report!* 🌟\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💲 *Current Price:* `{price:.2f}` USDT\n"
        f"📢 *The Verdict:* {signal_highlight}\n"
        f"🧠 *Confidence Score:* `{score}`\n"
        f"{sl_tp_text}\n\n"
        f"🔬 *Detailed Analysis:*\n"
        f"{reasons_text}\n\n"
        f"{news_section}\n\n" # Insert the combined news section here
        f"⏰ *Last Updated:* `{ist_time}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_💡 Remember: Crypto is volatile. Always do your own research and manage your risks._"
    )

    if signal in ["BUY", "SELL"] and SUCCESS_STICKER_ID:
        try:
            await context.bot.send_sticker(chat_id=chat_id, sticker=SUCCESS_STICKER_ID)
        except Exception as e:
            logger.error(f"Failed to send success sticker: {e}")
    elif signal == "WAIT" and ERROR_STICKER_ID:
        try:
            await context.bot.send_sticker(chat_id=chat_id, sticker=ERROR_STICKER_ID)
        except Exception as e:
            logger.error(f"Failed to send error sticker: {e}")

    await query.edit_message_text(
        final_message,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Refresh Signal ♻️", callback_data="signal")]])
    )

# --- Main Runner ---
def main():
    """Starts the Telegram bot."""
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.run_polling()

if __name__ == '__main__':
    main()
