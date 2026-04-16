"""Befehlserkennung via spaCy + rapidfuzz (basiert auf Betreuer-Vorlage CommandProcess.py)."""

import spacy
from rapidfuzz import fuzz, process
from .mill_commands import MillCommands

class CommandProcess:
    nlp = spacy.load("de_core_news_sm") # python -m spacy download de_core_news_sm

    def __init__(self,text_q_in, text_q_out, list_cmd: MillCommands):
        self.text_q_in = text_q_in
        self.text_q_out = text_q_out
        self.list = list_cmd.cmd
        self.keys = list(list_cmd.cmd.keys())

    def find_match(self,full_sentence):
        doc = self.nlp(full_sentence)
        lemmata = [token.lemma_ for token in doc]
            # Hotword-Erkennung mit hotword_list
            # Und vergleich mit lemmata
        highestScore, highestMatch = 0, None
        for word in full_sentence.split():
            result = process.extractOne(word, self.keys, scorer=fuzz.ratio)
            if result is None:
                continue
            match, score, _ = result
            if score > highestScore:
                highestScore = score
                highestMatch = match
        for lem in lemmata:
            result_lem = process.extractOne(lem, self.keys, scorer=fuzz.ratio)
            if result_lem is None:
                continue
            match_lem, score_lem, _ = result_lem
            if score_lem > highestScore:
                highestScore = score_lem
                highestMatch = match_lem
        if highestScore>70:
            #print(f"\nMatch: {highestMatch}")
            return highestMatch
        else:
            return None

    def find_all_matches(self, full_sentence):
        """Wie find_match(), aber gibt ALLE Treffer > 70 zurueck (fuer Multi-Positions-Zuege)."""
        doc = self.nlp(full_sentence)
        lemmata = [token.lemma_ for token in doc]
        found = []
        for word in full_sentence.split():
            result = process.extractOne(word, self.keys, scorer=fuzz.ratio)
            if result is None:
                continue
            match, score, _ = result
            if score > 70 and match not in found:
                found.append(match)
        for lem in lemmata:
            result = process.extractOne(lem, self.keys, scorer=fuzz.ratio)
            if result is None:
                continue
            match, score, _ = result
            if score > 70 and match not in found:
                found.append(match)
        return found if found else None

    def process_sentence(self):
        while True:
            text = self.text_q_in.get()
            matches = self.find_all_matches(text)
            if matches is not None:
                self.text_q_out.put(matches)
