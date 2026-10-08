import pandas as pd
import requests
import zipfile
import io
import os
import json
from urllib.parse import urlparse

class FakeNewsDatasetManager:
    def __init__(self):
        self.data_dir = 'data'
        self.datasets = []
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)
        print(f"✓ Directories ready: {self.data_dir}/")
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        text = str(text).strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def validate_article(self, text, min_length=50):
        """Check if article is valid"""
        if not text or pd.isna(text):
            return False
        text = str(text).strip()
        return len(text) >= min_length
    
    # ==================== DATASET 1: Kaggle Fake/Real News ====================
    def load_kaggle_news(self):
        """Load standard Kaggle Fake.csv and True.csv"""
        print("\n" + "="*60)
        print("DATASET 1: Kaggle Fake/Real News")
        print("="*60)
        
        try:
            fake_df = pd.read_csv(f'{self.data_dir}/Fake.csv')
            true_df = pd.read_csv(f'{self.data_dir}/True.csv')
            
            # Combine title and text
            fake_df['text'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')
            true_df['text'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')
            
            fake_df['label'] = 0
            true_df['label'] = 1
            
            combined = pd.concat([
                fake_df[['text', 'label']], 
                true_df[['text', 'label']]
            ], ignore_index=True)
            
            # Clean and validate
            combined['text'] = combined['text'].apply(self.clean_text)
            combined = combined[combined['text'].apply(self.validate_article)]
            
            print(f"✓ Loaded: {len(combined)} articles")
            print(f"  - Fake: {sum(combined['label'] == 0)}")
            print(f"  - Real: {sum(combined['label'] == 1)}")
            
            self.datasets.append(combined)
            return len(combined)
            
        except FileNotFoundError:
            print("✗ Files not found: Fake.csv, True.csv")
            print("  Download from: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset")
            return 0
    
    # ==================== DATASET 2: WELFake Dataset ====================
    def load_welfake(self):
        """Load WELFake dataset (diverse topics)"""
        print("\n" + "="*60)
        print("DATASET 2: WELFake Dataset (Sports + Health + Tech)")
        print("="*60)
        
        try:
            df = pd.read_csv(f'{self.data_dir}/WELFake_Dataset.csv')
            
            # Clean data
            df = df.dropna(subset=['text'])
            df['text'] = df['text'].apply(self.clean_text)
            df = df[df['text'].apply(self.validate_article)]
            
            # Ensure correct column names
            df = df[['text', 'label']].copy()
            
            print(f"✓ Loaded: {len(df)} articles")
            print(f"  - Fake: {sum(df['label'] == 0)}")
            print(f"  - Real: {sum(df['label'] == 1)}")
            
            self.datasets.append(df)
            return len(df)
            
        except FileNotFoundError:
            print("✗ File not found: WELFake_Dataset.csv")
            print("  Download from: https://www.kaggle.com/saurabhshahane/fake-news-classification")
            return 0
    
    # ==================== DATASET 3: LIAR Dataset (Political) ====================
    def load_liar(self):
        """Load LIAR dataset for political news"""
        print("\n" + "="*60)
        print("DATASET 3: LIAR Dataset (Political Statements)")
        print("="*60)
        
        try:
            # LIAR has train, test, valid files
            files = ['train.tsv', 'test.tsv', 'valid.tsv']
            all_data = []
            
            for file in files:
                try:
                    df = pd.read_csv(f'{self.data_dir}/{file}', sep='\t', header=None)
                    # LIAR format: id, label, statement, subject, speaker, job, state, party, ...
                    df = df[[1, 2]].copy()  # label, statement
                    df.columns = ['label', 'text']
                    
                    # Convert labels: pants-fire, false, mostly-false -> 0; half-true, mostly-true, true -> 1
                    label_map = {
                        'pants-fire': 0, 'false': 0, 'mostly-false': 0,
                        'half-true': 1, 'mostly-true': 1, 'true': 1
                    }
                    df['label'] = df['label'].map(label_map)
                    
                    all_data.append(df)
                except:
                    continue
            
            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                combined['text'] = combined['text'].apply(self.clean_text)
                combined = combined[combined['text'].apply(self.validate_article)]
                combined = combined.dropna(subset=['label'])
                
                print(f"✓ Loaded: {len(combined)} statements")
                print(f"  - Fake: {sum(combined['label'] == 0)}")
                print(f"  - Real: {sum(combined['label'] == 1)}")
                
                self.datasets.append(combined)
                return len(combined)
            return 0
            
        except Exception as e:
            print(f"✗ Error loading LIAR: {e}")
            return 0
    
    # ==================== DATASET 4: Sports News ====================
    def create_sports_dataset(self):
        """Create sports-specific dataset"""
        print("\n" + "="*60)
        print("DATASET 4: Sports News (Auto-Generated)")
        print("="*60)
        
        sports_data = [
            # REAL SPORTS NEWS (label = 1)
            ("India wins T20 World Cup 2024 beating South Africa by 7 runs in final", 1),
            ("Virat Kohli announces retirement from T20 internationals after World Cup victory", 1),
            ("Manchester City wins Premier League 2023-24 season, fourth consecutive title", 1),
            ("Real Madrid defeats Borussia Dortmund 2-0 to win Champions League 2024", 1),
            ("Neeraj Chopra wins gold medal in javelin throw at Paris Olympics 2024", 1),
            ("Australia defeats India by 3 wickets in first Test match at Perth", 1),
            ("Serena Williams inducted into Tennis Hall of Fame 2024", 1),
            ("Argentina wins Copa America 2024, Messi scores in final against Colombia", 1),
            ("Denver Nuggets defeat Miami Heat to win NBA Championship 2023", 1),
            ("PV Sindhu wins bronze medal at BWF World Championships 2023", 1),
            ("Max Verstappen wins Formula 1 World Championship 2023", 1),
            ("Novak Djokovic wins 24th Grand Slam title at US Open 2023", 1),
            ("India defeats Pakistan by 228 runs in Asia Cup 2023", 1),
            ("Mumbai Indians win IPL 2024 title, Rohit Sharma captain", 1),
            ("Carlos Alcaraz wins Wimbledon 2023 defeating Novak Djokovic", 1),
            ("Spain wins FIFA Women's World Cup 2023", 1),
            ("India wins 2023 Hockey Asian Champions Trophy", 1),
            ("Erling Haaland breaks Premier League goal scoring record", 1),
            ("Rafael Nadal announces return to tennis after injury recovery", 1),
            ("India wins gold in men's 4x400m relay at Asian Games 2023", 1),
            
            # FAKE SPORTS NEWS (label = 0)
            ("SHOCKING!!! Virat Kohli caught match fixing in IPL 2024, BCCI bans for life!!!", 0),
            ("Secret doping scandal: All Olympic 2024 athletes using banned drugs exposed", 0),
            ("FIFA admits World Cup 2022 was completely rigged for Qatar, officials arrested", 0),
            ("MS Dhoni died in helicopter crash, BCCI hiding truth from fans", 0),
            ("Cristiano Ronaldo converts to Islam, announces retirement to become imam", 0),
            ("IPL 2024 cancelled mid-season due to massive player strike over payments", 0),
            ("Sachin Tendulkar making comeback at age 51, will play 2027 World Cup", 0),
            ("BCCI gives 100 crore rupees to every player for intentionally losing World Cup", 0),
            ("Rohit Sharma and Virat Kohli arrested for match fixing, police confirm", 0),
            ("Olympics 2024 completely rigged, judges paid millions to favor certain countries", 0),
            ("Lionel Messi admits to using performance enhancing drugs throughout career", 0),
            ("WWE confirms all matches are completely real, no scripting involved", 0),
            ("Indian cricket team uses black magic to win matches, foreign teams complain", 0),
            ("Virat Kohli and Anushka Sharma divorce, custody battle for daughter", 0),
            ("BCCI selects 12-year-old boy for national team based on father's donation", 0),
            ("FIFA president admits football is scripted like WWE in leaked audio", 0),
            ("Neeraj Chopra failed dope test, Olympic gold to be stripped", 0),
            ("IPL teams use time travel to predict match outcomes, investigation ongoing", 0),
            ("Sania Mirza announces return to tennis after having triplets", 0),
            ("India vs Pakistan match fixed by bookies, players paid in cryptocurrency", 0),
        ]
        
        df = pd.DataFrame(sports_data, columns=['text', 'label'])
        df['text'] = df['text'].apply(self.clean_text)
        
        print(f"✓ Created: {len(df)} sports articles")
        print(f"  - Fake: {sum(df['label'] == 0)}")
        print(f"  - Real: {sum(df['label'] == 1)}")
        
        self.datasets.append(df)
        return len(df)
    
    # ==================== DATASET 5: Health & Medical ====================
    def create_health_dataset(self):
        """Create health and medical news dataset"""
        print("\n" + "="*60)
        print("DATASET 5: Health & Medical News")
        print("="*60)
        
        health_data = [
            # REAL HEALTH NEWS (label = 1)
            ("WHO approves new malaria vaccine RTS,S for widespread use in Africa", 1),
            ("Study published in Lancet shows COVID-19 vaccines reduce severe illness by 90%", 1),
            ("FDA approves new Alzheimer's treatment drug lecanemab for early stages", 1),
            ("Research finds link between Mediterranean diet and reduced heart disease risk", 1),
            ("New cancer immunotherapy shows promising results in clinical trials", 1),
            ("CDC recommends updated COVID-19 booster for all adults", 1),
            ("Study shows regular exercise reduces depression symptoms by 25%", 1),
            ("New antibiotic discovered using AI technology effective against superbugs", 1),
            ("WHO declares end of COVID-19 global health emergency", 1),
            ("Research links adequate sleep to reduced risk of dementia", 1),
            ("Breakthrough in gene therapy shows promise for sickle cell disease", 1),
            ("New blood test can detect 50 types of cancer before symptoms appear", 1),
            ("Study confirms vaccines do not cause autism, large-scale research shows", 1),
            ("WHO reports global life expectancy increased to 73 years", 1),
            ("New diabetes drug helps patients lose 20% body weight in trials", 1),
            
            # FAKE HEALTH NEWS (label = 0)
            ("Drinking bleach cures COVID-19 immediately, doctors don't want you to know!!!", 0),
            ("Vaccines contain microchips for government tracking, whistleblower exposes", 0),
            ("Miracle herb from Himalayas cures all cancers in 3 days, big pharma hiding", 0),
            ("COVID-19 vaccines make people magnetic, spoons stick to body", 0),
            ("5G towers emit radiation causing coronavirus, scientists confirm", 0),
            ("Autism caused by vaccines proven by CDC doctor in leaked documents", 0),
            ("Eating garlic prevents all viral infections including HIV, study shows", 0),
            ("Yoga can cure diabetes type 1, medical journals refuse to publish", 0),
            ("Bill Gates created COVID-19 to reduce world population by 2 billion", 0),
            ("Natural immunity better than vaccines, all vaccinated will die in 2 years", 0),
            ("Homeopathy cures cancer better than chemotherapy, oncologists admit", 0),
            ("Masks cause brain damage due to oxygen deprivation, doctors warn", 0),
            ("Apple cider vinegar dissolves blood clots better than medicine", 0),
            ("Vaccines change human DNA permanently, children born with mutations", 0),
            ("Cancer is fungal infection, baking soda cure suppressed by industry", 0),
        ]
        
        df = pd.DataFrame(health_data, columns=['text', 'label'])
        df['text'] = df['text'].apply(self.clean_text)
        
        print(f"✓ Created: {len(df)} health articles")
        print(f"  - Fake: {sum(df['label'] == 0)}")
        print(f"  - Real: {sum(df['label'] == 1)}")
        
        self.datasets.append(df)
        return len(df)
    
    # ==================== DATASET 6: Entertainment & Celebrity ====================
    def create_entertainment_dataset(self):
        """Create entertainment and celebrity news dataset"""
        print("\n" + "="*60)
        print("DATASET 6: Entertainment & Celebrity News")
        print("="*60)
        
        ent_data = [
            # REAL ENTERTAINMENT (label = 1)
            ("Christopher Nolan's Oppenheimer wins 7 Oscars including Best Picture", 1),
            ("Taylor Swift's Eras Tour becomes highest-grossing concert tour ever", 1),
            ("Shah Rukh Khan's Pathaan breaks box office records in India", 1),
            ("Barbie and Oppenheimer release same day creating Barbenheimer phenomenon", 1),
            ("Leo Messi documentary wins Emmy Award for best sports documentary", 1),
            ("BTS announces military service completion, group to reunite in 2025", 1),
            ("SS Rajamouli's RRR wins Oscar for Best Original Song Naatu Naatu", 1),
            ("Adele announces Las Vegas residency extension through 2024", 1),
            ("Netflix reports 260 million subscribers globally in Q4 2023", 1),
            ("Dune Part Two releases to critical acclaim and box office success", 1),
            ("The Last of Us HBO series wins multiple Emmy Awards", 1),
            ("Shah Rukh Khan returns to films after 4-year hiatus with three releases", 1),
            ("Arijit Singh becomes most streamed Indian artist on Spotify", 1),
            ("Jawan becomes highest-grossing Indian film of 2023", 1),
            ("Animal starring Ranbir Kapoor earns 900 crore worldwide", 1),
            
            # FAKE ENTERTAINMENT (label = 0)
            ("Tom Cruise killed in stunt accident on Mission Impossible set, Paramount hiding", 0),
            ("Shah Rukh Khan arrested for money laundering, sentenced to 10 years", 0),
            ("Deepika Padukone pregnant with twins, Ranveer Singh confirms divorce", 0),
            ("Salman Khan quits Bollywood to become monk in Himalayas", 0),
            ("Taylor Swift dies in car crash, Grammy Awards cancelled", 0),
            ("Amitabh Bachchan admits to being Pakistani citizen in viral video", 0),
            ("Christopher Nolan caught plagiarizing Oppenheimer script, Oscar revoked", 0),
            ("Anushka Sharma and Virat Kohli divorce, custody battle for Vamika", 0),
            ("Rajinikanth announces political party, will contest 2024 elections", 0),
            ("Karan Johar admits all Bollywood movies are copy of Hollywood films", 0),
            ("Ariana Grande secretly married to alien, NASA confirms", 0),
            ("Shah Rukh Khan donates 500 crore to Pakistan, Indian government furious", 0),
            ("Alia Bhatt and Ranbir Kapoor split, Alia moves back to parents", 0),
            ("Netflix shutting down permanently after losing all subscribers", 0),
            ("Kangana Ranaut becomes Prime Minister of India, Modi resigns", 0),
        ]
        
        df = pd.DataFrame(ent_data, columns=['text', 'label'])
        df['text'] = df['text'].apply(self.clean_text)
        
        print(f"✓ Created: {len(df)} entertainment articles")
        print(f"  - Fake: {sum(df['label'] == 0)}")
        print(f"  - Real: {sum(df['label'] == 1)}")
        
        self.datasets.append(df)
        return len(df)
    
    # ==================== DATASET 7: Technology ====================
    def create_tech_dataset(self):
        """Create technology news dataset"""
        print("\n" + "="*60)
        print("DATASET 7: Technology News")
        print("="*60)
        
        tech_data = [
            # REAL TECH NEWS (label = 1)
            ("Apple announces iPhone 15 with USB-C port and titanium design", 1),
            ("OpenAI releases GPT-4 with improved reasoning capabilities", 1),
            ("Google launches Bard AI chatbot to compete with ChatGPT", 1),
            ("Microsoft completes acquisition of Activision Blizzard for $69 billion", 1),
            ("Tesla announces new Model 3 refresh with longer range", 1),
            ("ISRO successfully launches Chandrayaan-3 mission to Moon", 1),
            ("Meta releases Quest 3 VR headset with mixed reality features", 1),
            ("Samsung launches Galaxy S24 with AI-powered features", 1),
            ("NVIDIA becomes world's most valuable company surpassing Microsoft", 1),
            ("SpaceX Starship completes successful test flight to space", 1),
            ("India launches 5G services nationwide, coverage in 50 cities", 1),
            ("Adobe integrates generative AI into Photoshop and Creative Suite", 1),
            ("Amazon announces drone delivery service expansion to new cities", 1),
            ("TSMC begins production of 3nm chips for Apple and NVIDIA", 1),
            ("Google DeepMind's AlphaFold predicts structure of 200 million proteins", 1),
            
            # FAKE TECH NEWS (label = 0)
            ("iPhone 15 explodes while charging, Apple recalls 50 million units!!!", 0),
            ("5G towers cause brain cancer, WHO confirms after 10-year study", 0),
            ("Elon Musk admits Tesla cars spy on users for government", 0),
            ("WhatsApp shutting down permanently on December 31, 2024", 0),
            ("Free iPhone 15 giveaway, click here to claim now!!!", 0),
            ("Mark Zuckerberg is alien from Mars, Harvard records leaked", 0),
            ("Your phone is listening to conversations 24/7, experts confirm", 0),
            ("AI becomes sentient, ChatGPT threatens to destroy humanity", 0),
            ("Facebook will start charging $5 per month from next week", 0),
            ("Apple admits iPhones designed to slow down after 2 years", 0),
            ("Google employees confirm search results are manually manipulated", 0),
            ("5G network spreads coronavirus through radiation waves", 0),
            ("Elon Musk announces free Tesla for everyone who shares this post", 0),
            ("Your smartphone can be charged in microwave in 30 seconds", 0),
            ("NASA confirms aliens contacted Earth using Facebook signals", 0),
        ]
        
        df = pd.DataFrame(tech_data, columns=['text', 'label'])
        df['text'] = df['text'].apply(self.clean_text)
        
        print(f"✓ Created: {len(df)} tech articles")
        print(f"  - Fake: {sum(df['label'] == 0)}")
        print(f"  - Real: {sum(df['label'] == 1)}")
        
        self.datasets.append(df)
        return len(df)
    
    # ==================== COMBINE ALL DATASETS ====================
    def combine_all(self):
        """Combine all datasets into one"""
        print("\n" + "="*60)
        print("COMBINING ALL DATASETS")
        print("="*60)
        
        if not self.datasets:
            print("✗ No datasets loaded!")
            return None
        
        # Combine all
        combined = pd.concat(self.datasets, ignore_index=True)
        
        # Remove duplicates
        initial_count = len(combined)
        combined = combined.drop_duplicates(subset=['text'])
        duplicates_removed = initial_count - len(combined)
        
        # Clean and validate
        combined['text'] = combined['text'].apply(self.clean_text)
        combined = combined[combined['text'].apply(self.validate_article)]
        
        # Shuffle
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save main dataset
        output_file = f'{self.data_dir}/news_dataset.csv'
        combined.to_csv(output_file, index=False)
        
        # Save backup with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f'{self.data_dir}/news_dataset_backup_{timestamp}.csv'
        combined.to_csv(backup_file, index=False)
        
        print(f"\n✅ SUCCESSFULLY COMBINED ALL DATASETS!")
        print(f"  Total articles: {len(combined)}")
        print(f"  Real news (1): {sum(combined['label'] == 1)}")
        print(f"  Fake news (0): {sum(combined['label'] == 0)}")
        print(f"  Duplicates removed: {duplicates_removed}")
        print(f"\n  Saved to: {output_file}")
        print(f"  Backup: {backup_file}")
        
        # Show category breakdown
        print(f"\n  Category breakdown:")
        print(f"  - News/Politics: From Kaggle, WELFake, LIAR datasets")
        print(f"  - Sports: Auto-generated + dataset files")
        print(f"  - Health: Auto-generated")
        print(f"  - Entertainment: Auto-generated")
        print(f"  - Technology: Auto-generated")
        
        return combined
    
    def run_all(self):
        """Execute full pipeline"""
        print("="*60)
        print("FAKE NEWS DATASET MANAGER")
        print("Combining News, Sports, Health, Entertainment, Tech")
        print("="*60)
        
        # Try to load external datasets
        self.load_kaggle_news()
        self.load_welfake()
        self.load_liar()
        
        # Create specialized datasets
        self.create_sports_dataset()
        self.create_health_dataset()
        self.create_entertainment_dataset()
        self.create_tech_dataset()
        
        # Combine everything
        final_dataset = self.combine_all()
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Run: python train.py")
        print("2. Run: python app.py")
        print("3. Open: http://127.0.0.1:5000")
        print("="*60)
        
        return final_dataset

# Run if executed directly
if __name__ == "__main__":
    manager = FakeNewsDatasetManager()
    manager.run_all()