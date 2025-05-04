# MudraSpeek Gemini AI Integration Guide

This guide explains how to use the new Gemini AI-powered features in the ISL (Indian Sign Language) recognition system.

## Setup

1. First, you need to get a Gemini API key:
   - Go to https://ai.google.dev/
   - Sign up or sign in
   - Create an API key

2. Set up your API key using one of these methods:
   - **Option 1:** Create a file named `gemini_api_key.txt` in the same directory as the application and paste your API key there
   - **Option 2:** Set it as an environment variable:
     - Windows: `set GEMINI_API_KEY=your_api_key_here`
     - Linux/Mac: `export GEMINI_API_KEY=your_api_key_here`

3. Install the required libraries:
   ```
   pip install google-generativeai gtts playsound
   ```

## New Features

### 1. Word Suggestions

As you sign individual letters, the system will now:
- Build words letter by letter
- Use Gemini AI to suggest potential words based on the letters you've signed
- Allow you to select from suggested words using keyboard shortcuts

**How to use word suggestions:**
- Sign letters one by one (A, B, C, etc.)
- Look at the "WORD SUGGESTIONS" section
- Press keys 1-3 to highlight one of the suggestions
- Press 'w' to select the highlighted suggestion and add it to your sentence

### 2. Sentence Processing and Text-to-Speech

The system can now:
- Process complete sentences to improve grammar and coherence
- Speak sentences aloud using text-to-speech
- **Auto-speak** words or sentences after 3 seconds of no gesture detection

**How to use sentence processing and speech:**
- After forming a sentence, press 's' to speak it manually
- Or simply wait 3 seconds without making any gestures to auto-speak:
  - If you've been typing letters, the current word will be spoken and added to the sentence
  - If you're between words, the entire sentence will be spoken
- If Gemini AI is enabled, it will process the text to improve grammar and coherence
- Both the original and AI-processed sentences will be displayed

## Keyboard Controls

- `s`: Speak the current sentence (with AI processing if enabled)
- `1`, `2`, `3`: Select a suggested word (highlight it)
- `w`: Use the currently selected word suggestion
- `space`: Complete current word and add a space
- `backspace`: Delete the last letter or word
- `c`: Clear the entire sentence
- `q`: Quit the recognition window

## Troubleshooting

If you encounter issues:

1. Check that your API key is correctly set up
2. Ensure you have internet connectivity for Gemini API calls
3. If text-to-speech isn't working, ensure your system has audio capabilities
4. For audio playback issues, make sure the playsound library is installed:
   ```
   pip install playsound
   ```

For more information about Gemini AI, visit: https://ai.google.dev/docs 