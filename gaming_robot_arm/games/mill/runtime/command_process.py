"""Erkennt Muehle-Brettlabel in gesprochenem Text mit spaCy-Lemmatisierung und rapidfuzz-Fuzzy-Matching."""

import re
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

    def _normalize(self, text: str) -> str:
        """Ersetzt Ring-Aliase und deutsche Zahlwörter, kombiniert dann 'B 3' -> 'B3'."""
        result = text
        for alias, letter in MillCommands.LETTER_ALIASES.items():
            result = re.sub(rf'\b{re.escape(alias)}\b', letter, result, flags=re.IGNORECASE)
        for alias, ring in MillCommands.RING_ALIASES.items():
            result = re.sub(rf'\b{re.escape(alias)}\b', ring, result, flags=re.IGNORECASE)
        for word, n in MillCommands.GERMAN_NUMBERS.items():
            result = re.sub(rf'\b{re.escape(word)}\b', str(n), result, flags=re.IGNORECASE)
        result = re.sub(r'\b([ABC])\W+([1-8])\b', r'\1\2', result, flags=re.IGNORECASE)
        return result

    def find_match(self, full_sentence):
        full_sentence = self._normalize(full_sentence)
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
        full_sentence = self._normalize(full_sentence)
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

    def find_undo(self, full_sentence: str) -> bool:
        """Erkennt Undo-Schluesselwoerter ueber Fuzzy-Match (Schwelle 80)."""
        norm = self._normalize(full_sentence).lower()
        for word in norm.split():
            result = process.extractOne(word, MillCommands.UNDO_KEYWORDS, scorer=fuzz.ratio)
            if result is not None and result[1] > 80:
                return True
        return False

    def find_confirm(self, full_sentence: str):
        """Erkennt Bestaetigung; True=ja, False=nein, None=unklar."""
        norm = self._normalize(full_sentence).lower()
        for word in norm.split():
            for kw in MillCommands.CONFIRM_YES:
                if fuzz.ratio(word, kw) > 80:
                    return True
            for kw in MillCommands.CONFIRM_NO:
                if fuzz.ratio(word, kw) > 80:
                    return False
        return None

    def process_sentence(self):
        while True:
            text = self.text_q_in.get()
            if self.find_undo(text):
                self.text_q_out.put(["__UNDO__"])
                continue
            confirm = self.find_confirm(text)
            if confirm is True:
                self.text_q_out.put(["__YES__"])
                continue
            if confirm is False:
                self.text_q_out.put(["__NO__"])
                continue
            matches = self.find_all_matches(text)
            if matches is not None:
                self.text_q_out.put(matches)
