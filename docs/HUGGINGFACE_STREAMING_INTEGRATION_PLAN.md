# HuggingFace Streaming Integration Plan

## Objective
Expand TRM input streams from 10 to 20+ features by integrating real-time market sentiment, news analysis, and social media signals via HuggingFace models.

## Current State (Verified ✅)
- **10 input streams** processing correctly:
  1. VIX volatility
  2-3. SPY momentum (5d, 20d)
  4-6. Volume/breadth/correlation
  7-10. Advanced metrics (put/call, Gini, dispersion, quality)

- **Recursive training loop active**: T=3 × n=6 = 42 effective layers
- **All features impact output**: 0.15-0.35 per feature

## Proposed Expansion: +12 HuggingFace Streams

### New Input Streams (11-22)

**Sentiment Analysis (Streams 11-13)**
- Stream 11: **News Sentiment** - FinBERT sentiment on financial news
- Stream 12: **Social Sentiment** - Twitter/Reddit sentiment (RoBERTa)
- Stream 13: **Earnings Call Tone** - Transcript sentiment analysis

**Semantic Features (Streams 14-16)**
- Stream 14: **Market Regime Embedding** - BERT encoding of market conditions
- Stream 15: **Sector Narrative** - Semantic similarity to historical patterns
- Stream 16: **Risk Embedding** - GPT-2 generated risk assessment

**Time Series Forecasting (Streams 17-19)**
- Stream 17: **Price Forecast** - TimeGPT 1-day ahead prediction
- Stream 18: **Volatility Forecast** - Transformer-based vol prediction
- Stream 19: **Correlation Forecast** - Predicted market correlation

**Alternative Data (Streams 20-22)**
- Stream 20: **Supply Chain Sentiment** - Logistics/shipping indicators
- Stream 21: **Geopolitical Risk** - News embedding for geopolitical events
- Stream 22: **Consumer Confidence** - Social media spending signals

## Architecture Changes

### 1. Expanded Feature Extractor

```python
# src/data/hf_feature_extractor.py
class HuggingFaceFeatureExtractor:
    def __init__(self):
        # Load HuggingFace models
        self.finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

        self.time_gpt = TimeGPT(api_key=os.getenv('TIMEGPT_API_KEY'))
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_sentiment_features(self, date: str) -> Dict[str, float]:
        """Extract streams 11-13: Sentiment signals"""
        # Fetch news for date
        news = self.fetch_news(date)

        # FinBERT sentiment
        inputs = self.finbert_tokenizer(news, return_tensors="pt", truncation=True)
        outputs = self.finbert(**inputs)
        sentiment_score = torch.softmax(outputs.logits, dim=1)[0][2].item()  # Positive class

        return {
            'news_sentiment': sentiment_score,
            'social_sentiment': self.get_social_sentiment(date),
            'earnings_tone': self.get_earnings_tone(date)
        }
```

### 2. Streaming Data Pipeline

```python
# src/data/streaming_pipeline.py
class MarketDataStreamer:
    def __init__(self):
        self.hf_extractor = HuggingFaceFeatureExtractor()
        self.market_extractor = MarketFeatureExtractor()

    async def stream_features(self):
        """Real-time streaming of all 22 features"""
        while True:
            current_date = datetime.now()

            # Parallel extraction
            market_features = await asyncio.create_task(
                self.market_extractor.extract_features_realtime()
            )

            hf_features = await asyncio.create_task(
                self.hf_extractor.extract_all_features()
            )

            # Combine into 22-dimensional vector
            combined = np.concatenate([market_features, hf_features])

            yield combined

            await asyncio.sleep(60)  # 1-minute intervals
```

### 3. Updated TRM Model

```python
# Expand input_dim from 10 to 22
model = TinyRecursiveModel(
    input_dim=22,  # Was 10
    hidden_dim=512,
    output_dim=8,
    num_latent_steps=6,
    num_recursion_cycles=3
)
```

### 4. HuggingFace Hub Integration

```python
# src/integrations/hf_hub.py
from huggingface_hub import HfApi, Repository

class TRMHuggingFaceIntegration:
    def __init__(self, repo_name="trader-ai-trm"):
        self.api = HfApi()
        self.repo_name = repo_name

    def push_model_to_hub(self, model_path: str):
        """Upload trained TRM to HuggingFace Hub"""
        self.api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="trm_checkpoint.pkl",
            repo_id=f"your-org/{self.repo_name}",
            repo_type="model"
        )

    def stream_predictions(self):
        """Stream live TRM predictions to HuggingFace Spaces"""
        # Create gradio interface
        import gradio as gr

        def predict(features):
            # Load model and predict
            prediction = self.model.predict(features)
            return prediction

        iface = gr.Interface(
            fn=predict,
            inputs=gr.Dataframe(headers=["Feature {}".format(i) for i in range(22)]),
            outputs=gr.Label(num_top_classes=8)
        )

        iface.launch()
```

## Implementation Phases

### Phase 1: HuggingFace Model Integration (Week 1)
- [ ] Install transformers, sentence-transformers, timegpt
- [ ] Implement FinBERT sentiment extraction
- [ ] Implement social media sentiment (RoBERTa)
- [ ] Test sentiment features on historical data

### Phase 2: Expanded Feature Pipeline (Week 2)
- [ ] Create HuggingFaceFeatureExtractor class
- [ ] Implement 12 new feature streams
- [ ] Add caching layer for HF API calls
- [ ] Validate feature quality

### Phase 3: Streaming Infrastructure (Week 3)
- [ ] Implement async streaming pipeline
- [ ] Add Redis queue for feature buffering
- [ ] Create WebSocket server for real-time delivery
- [ ] Test streaming performance (target: <1s latency)

### Phase 4: TRM Model Expansion (Week 4)
- [ ] Retrain TRM with 22 input features
- [ ] Adjust normalization for new features
- [ ] Validate recursive loop still works
- [ ] Benchmark performance vs 10-feature model

### Phase 5: HuggingFace Hub Deployment (Week 5)
- [ ] Create HuggingFace Space for live predictions
- [ ] Upload model to HuggingFace Hub
- [ ] Create interactive Gradio demo
- [ ] Add model card documentation

## Expected Improvements

**With 22 Input Streams:**
- **Better market regime detection**: Sentiment captures narratives
- **Earlier crisis signals**: Social/news sentiment leads price
- **Improved accuracy**: More information → better decisions
- **Target**: 70%+ validation accuracy (vs current 57%)

## Technical Requirements

### Dependencies
```bash
pip install transformers sentence-transformers timegpt-py
pip install datasets huggingface-hub gradio
pip install redis aioredis websockets
```

### API Keys Needed
- HuggingFace API token (free)
- TimeGPT API key (paid tier)
- Twitter API (optional, for social sentiment)
- News API key (free tier available)

### Compute Requirements
- **Training**: NVIDIA GPU with 16GB+ VRAM (for expanded model)
- **Inference**: 8GB VRAM sufficient
- **Streaming**: 4-core CPU, 16GB RAM for real-time processing

## Risk Mitigation

**Risk 1: Feature Extraction Latency**
- Mitigation: Pre-compute HF features, cache for 1 hour
- Fallback: Use last known values if API times out

**Risk 2: API Rate Limits**
- Mitigation: Self-host smaller models (FinBERT, RoBERTa)
- Fallback: Implement exponential backoff + queueing

**Risk 3: Overfitting on Noisy Signals**
- Mitigation: Separate validation on out-of-sample period
- Fallback: Feature ablation study to remove noisy streams

## Success Metrics

1. **Feature Quality**: Each HF stream has >0.05 impact on predictions
2. **Latency**: <5 seconds for full 22-feature extraction
3. **Accuracy**: >65% validation accuracy (vs 57% baseline)
4. **Diversity**: Model predicts >2 strategies in backtest
5. **Streaming**: 99%+ uptime for real-time pipeline

## Next Steps

**Immediate Actions:**
1. Install HuggingFace dependencies
2. Test FinBERT on sample financial news
3. Create prototype with 13 features (10 existing + 3 sentiment)
4. Validate prototype improves accuracy

**Long-term Vision:**
- 50+ input streams (add crypto, macro, options flow)
- Multi-modal inputs (charts, earnings call audio)
- Ensemble with other HF models (GPT-4 for narratives)
- Fully autonomous trading with HF-powered signals
