# LiaScript Feature-Erkennung: Muster-Dokumentation

Diese Dokumentation beschreibt alle Features, die in der Feature-Extraktion erkannt werden, zusammen mit den verwendeten Regex-Mustern.

---

## Multimedia-Features

### Video

- **Beschreibung:** LiaScript Movie-Embeds
- **Syntax:** `!?[alt](url)`
- **Regex-Muster:** `!\?\[`
- **Beispiel:**
  ```markdown
  !?[Video-Titel](https://www.youtube.com/watch?v=xyz)
  ```

### Audio

- **Beschreibung:** Audio-Einbettungen
- **Syntax:** `?[alt](url)`
- **Regex-Muster:** `(?<!\?)\?\[` (Negativer Lookbehind, um `??[` auszuschließen)
- **Beispiel:**
  ```markdown
  ?[Audio-Titel](https://example.com/audio.mp3)
  ```

### Webapp

- **Beschreibung:** Interaktive Embeds/Webapps
- **Syntax:** `??[alt](url)`
- **Regex-Muster:** `\?\?\[`
- **Beispiel:**
  ```markdown
  ??[Interaktive Simulation](https://example.com/simulation)
  ```

### Bilder

- **Beschreibung:** Standard Markdown-Bilder
- **Syntax:** `![alt](url)`
- **Regex-Muster:** `(?<!\?)\!\[.*?\]\(.*?\)` (schließt Videos `!?[` aus)
- **Beispiel:**
  ```markdown
  ![Beschreibung](https://example.com/bild.png)
  ```

---

## Quiz-Typen

### Text-Quiz

- **Beschreibung:** Freitext-Eingabe Quiz
- **Syntax:** `[[Lösungstext]]`
- **Regex-Muster:** `\[\[(?![Xx ]\]\]|\?\]\])[^\[\]]+\]\]` (schließt MC-Marker und Hints aus)
- **Zusätzlich:** MC-Quiz-Items am Zeilenanfang werden subtrahiert
- **Beispiel:**
  ```markdown
  Was ist die Hauptstadt von Deutschland?

  [[Berlin]]
  ```

### Single-Choice Quiz

- **Beschreibung:** Einfachauswahl
- **Syntax:** `[( )]` oder `[(X)]`
- **Regex-Muster:** `^\s*\[\([Xx ]\)\]` (am Zeilenanfang)
- **Beispiel:**
  ```markdown
  Welche Farbe hat der Himmel?

  [( )] Grün
  [(X)] Blau
  [( )] Rot
  ```

### Multiple-Choice Quiz

- **Beschreibung:** Mehrfachauswahl
- **Syntax:** `[[ ]]` oder `[[X]]`
- **Regex-Muster:** `^\s*\[\[[Xx ]\]\]` (am Zeilenanfang)
- **Beispiel:**
  ```markdown
  Welche sind Primzahlen?

  [[X]] 2
  [[X]] 3
  [[ ]] 4
  [[X]] 5
  ```

### Quiz-Hinweise

- **Beschreibung:** Hinweise für Quiz-Fragen
- **Syntax:** `[[?]]`
- **Regex-Muster:** `\[\[\?\]\]`
- **Beispiel:**
  ```markdown
  [[Berlin]]
  [[?]] Die Stadt liegt an der Spree
  ```

---

## Tabellen

### LiaScript Visualisierungs-Tabellen

- **Beschreibung:** Tabellen mit LiaScript-Datentyp-Annotationen für Charts/Visualisierungen
- **Erkennungsmuster:** HTML-Kommentar `-->` gefolgt von Tabelle
- **Regex-Muster:** `-->\s*\n\s*\|`
- **Beispiel:**
  ```markdown
  <!-- data-type="BarChart" -->
  | Jahr | Wert |
  |------|------|
  | 2020 | 100  |
  | 2021 | 150  |
  ```

### Standard Markdown-Tabellen

- **Beschreibung:** Reguläre Markdown-Tabellen
- **Regex-Muster:** `^\s*\|[\s\-:|]+\|\s*$` (Tabellen-Header-Separator)
- **Beispiel:**
  ```markdown
  | Spalte 1 | Spalte 2 |
  |----------|----------|
  | Wert 1   | Wert 2   |
  ```

---

## Code und Skripte

### Code-Blöcke

- **Beschreibung:** Fenced Code Blocks mit oder ohne Sprachangabe
- **Regex-Muster:** `^```[\w+]*\s*$` (am Zeilenanfang, mit optionaler Sprache)
- **Beispiel:**
  ````markdown
  ```python
  print("Hello World")
  ```
  ````

### Code-Projekte

- **Beschreibung:** LiaScript Code-Projekte (mehrere verknüpfte Dateien)
- **Syntax:** ``` ```language+ ``` (mit + Marker)
- **Regex-Muster:** `^```\w*\+` (am Zeilenanfang)
- **Beispiel:**
  ````markdown
  ```js +main.js
  import { helper } from './helper.js';
  console.log(helper());
  ```

  ```js helper.js
  export function helper() { return "Hello"; }
  ```
  ````

### Script-Tags

- **Beschreibung:** Inline JavaScript-Ausführung
- **Regex-Muster:** `<script[^>]*>.*?</script>` (case-insensitive, DOTALL)
- **Beispiel:**
  ```markdown
  <script>
    console.log("Hello");
  </script>
  ```

---

## Animation und TTS

### TTS Narrator-Fragmente

- **Beschreibung:** Text-to-Speech Kommentare (inline)
- **Syntax:** `--{{Nummer}}--` oder `--{{Start-Ende}}--` für Bereiche
- **Regex-Muster:** `--\{\{\d+(?:-\d+)?\}\}--`
- **Beispiele:**
  ```markdown
  --{{1}}--
  Dies wird bei Schritt 1 vorgelesen.

  --{{2-4}}--
  Dies wird von Schritt 2 bis 4 vorgelesen.
  ```

### TTS Narrator-Blöcke

- **Beschreibung:** Mehrzeilige TTS-Blöcke mit Stern-Begrenzung
- **Syntax:** `--{{Nummer}}--` gefolgt von `****` ... `****`
- **Regex-Muster:** `--\{\{\d+(?:-\d+)?\}\}--\s*\n\*{4,}`
- **Beispiel:**
  ```markdown
                 --{{1}}--
  ****************************

  Dieser gesamte Text wird bei Schritt 1 vorgelesen.
  Er kann mehrere Absätze enthalten.

  ****************************
  ```

### Animations-Fragmente

- **Beschreibung:** Schrittweise Einblendungen (inline)
- **Syntax:** `{{Nummer}}` oder `{{Start-Ende}}` für Bereiche
- **Regex-Muster:** `(?<!--)\{\{\d+(?:-\d+)?\}\}` (nicht von `--` vorangestellt)
- **Beispiele:**
  ```markdown
  {{1}} Dieser Text erscheint bei Schritt 1.

  {{2-4}} Dieser Text ist von Schritt 2 bis 4 sichtbar.

  {{1}}
  Dieser Absatz erscheint bei Schritt 1.
  ```

### Animations-Blöcke

- **Beschreibung:** Mehrzeilige animierte Blöcke mit Stern-Begrenzung
- **Syntax:** `{{Nummer}}` gefolgt von `****` ... `****`
- **Regex-Muster:** `\{\{\d+(?:-\d+)?\}\}\s*\n\*{4,}`
- **Beispiel:**
  ```markdown
                 {{1}}
  ****************************

  Dieser gesamte Block erscheint bei Schritt 1.
  Er kann mehrere Absätze, Listen, Bilder etc. enthalten.

  ****************************
  ```

### Animate.css Animationen

- **Beschreibung:** CSS-Animationsklassen
- **Regex-Muster:** `class=["'][^"']*\banimated\b|animate__\w+`
- **Beispiel:**
  ```markdown
  <div class="animated fadeIn">Animierter Inhalt</div>
  ```

---

## Header-Felder

Der Header-Block (zwischen `<!--` und `-->`) wird automatisch extrahiert. Die folgenden Muster werden nur innerhalb dieses Blocks gesucht.

### Import-Statements (Templates)

- **Beschreibung:** Importierte LiaScript-Templates
- **Syntax:** `import: URL` (auch mehrzeilig möglich)
- **Erkennung:** Zeilen im Header, die mit `import:` beginnen, gefolgt von HTTP-URLs. Fortsetzungszeilen mit URLs werden ebenfalls erfasst.
- **Beispiel:**
  ```markdown
  <!--
  author: Max Mustermann
  import: https://raw.githubusercontent.com/LiaTemplates/AVR8js/main/README.md
          https://raw.githubusercontent.com/LiaTemplates/coderunner/main/README.md
  version: 1.0
  -->
  ```

### Externe Skripte

- **Beschreibung:** JavaScript-Bibliotheken im Header
- **Syntax:** `script: URL` (auch mehrzeilig möglich)
- **Erkennung:** Zeilen im Header, die mit `script:` beginnen, gefolgt von HTTP-URLs. Fortsetzungszeilen mit URLs werden ebenfalls erfasst.
- **Beispiel:**
  ```markdown
  <!--
  author: Max Mustermann
  script: https://cdn.jsdelivr.net/npm/chart.js
          https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js
  -->
  ```

### Externe CSS

- **Beschreibung:** CSS-Stylesheets im Header
- **Syntax:** `link: URL` (auch mehrzeilig möglich)
- **Erkennung:** Zeilen im Header, die mit `link:` beginnen, gefolgt von HTTP-URLs. Fortsetzungszeilen mit URLs werden ebenfalls erfasst.
- **Beispiel:**
  ```markdown
  <!--
  author: Max Mustermann
  link: https://example.com/style.css
        https://example.com/theme.css
  -->
  ```

### Logo

- **Beschreibung:** Kurs-Logo im Header
- **Syntax:** `logo: URL`
- **Regex-Muster:** `^\s*logo\s*:` (im Header)
- **Beispiel:**
  ```markdown
  <!--
  logo: https://example.com/course-logo.png
  -->
  ```

### Icon

- **Beschreibung:** Favicon/Icon im Header
- **Syntax:** `icon: URL`
- **Regex-Muster:** `^\s*icon\s*:` (im Header)
- **Beispiel:**
  ```markdown
  <!--
  icon: https://example.com/favicon.ico
  -->
  ```

### Narrator-Konfiguration

- **Beschreibung:** TTS-Konfiguration im Header
- **Syntax:** `narrator: Stimme`
- **Regex-Muster:** `^\s*narrator\s*:` (im Header)
- **Beispiel:**
  ```markdown
  <!--
  narrator: Deutsch Female
  -->
  ```

---

## Spezial-Features

### Mathematische Formeln

#### Inline-Formeln

- **Beschreibung:** Formeln im Fließtext
- **Syntax:** `$ ... $`
- **Regex-Muster:** `(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)` (nach Entfernung von Code-Blöcken)
- **Hinweis:** Code-Blöcke und Inline-Code werden vor der Analyse entfernt, um False Positives ($PATH, $HOME) zu vermeiden
- **Beispiel:**
  ```markdown
  Die Formel $E = mc^2$ ist bekannt.
  ```

#### Display-Formeln

- **Beschreibung:** Abgesetzte Formeln
- **Syntax:** `$$ ... $$`
- **Regex-Muster:** `\$\$[^$]+\$\$` (DOTALL, nach Entfernung von Code-Blöcken)
- **Beispiel:**
  ```markdown
  $$
  \int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
  $$
  ```

---

## Neue Features (v2)

### QR-Codes

- **Beschreibung:** QR-Code-Generierung
- **Syntax:** `[qr-code](daten)` oder `[qr-code (optionen)](daten)`
- **Regex-Muster:** `\[qr-code\]|\[qr-code\s*\(` (case-insensitive)
- **Beispiel:**
  ```markdown
  [qr-code](https://liascript.github.io)
  ```

### Surveys/Umfragen

- **Beschreibung:** Likert-Skala und Umfrage-Elemente
- **Syntax:** `[(Text)]` (wobei Text nicht X oder Leerzeichen ist)
- **Regex-Muster:** `^\s*\[\([^Xx ][^\)]+\)\]` (am Zeilenanfang)
- **Beispiel:**
  ```markdown
  Wie zufrieden sind Sie?

  [(sehr zufrieden)]
  [(zufrieden)]
  [(neutral)]
  [(unzufrieden)]
  [(sehr unzufrieden)]
  ```

### Footnotes

- **Beschreibung:** Fußnoten-Referenzen und -Definitionen
- **Syntax:** `[^id]` für Referenz, `[^id]: Text` für Definition
- **Regex-Muster Referenz:** `\[\^[^\]]+\]`
- **Regex-Muster Definition:** `^\[\^[^\]]+\]:` (am Zeilenanfang)
- **Beispiel:**
  ```markdown
  Dies ist ein Text mit Fußnote[^1].

  [^1]: Die Erklärung zur Fußnote.
  ```

### Makro-Aufrufe

- **Beschreibung:** LiaScript Makro-Aufrufe (aus Templates)
- **Syntax:** `@makroname` oder `@makroname(...)`
- **Regex-Muster:** `(?<![a-zA-Z0-9.])@[a-zA-Z_][a-zA-Z0-9_]*(?:\s*\(|(?=\s|$|[^a-zA-Z0-9_@.]))`
- **Beispiel:**
  ```markdown
  @runCode

  @embed(https://example.com)
  ```

### Effekte

- **Beschreibung:** LiaScript Effekt-Annotationen
- **Syntax:** `<!-- effect="..." -->` oder `data-effect="..."`
- **Regex-Muster:** `effect\s*=\s*["\']|data-effect\s*=` (case-insensitive)
- **Beispiel:**
  ```markdown
  <!-- effect="bounce" -->
  Dieser Text springt.
  ```

### Classroom-Features

- **Beschreibung:** Kollaborative Classroom-Funktionen
- **Syntax:** `@classroom`
- **Regex-Muster:** `@classroom` (case-insensitive)
- **Beispiel:**
  ```markdown
  @classroom
  ```

### ASCII-Diagramme

- **Beschreibung:** ASCII-Art und Diagramme
- **Syntax:** ``` ```ascii ```, ``` ```diagram ```, ``` ```svgbob ```
- **Regex-Muster:** `^```\s*(ascii|diagram|art|svgbob)\s*$` (case-insensitive)
- **Beispiel:**
  ````markdown
  ```ascii
  +-------+     +-------+
  | Start |---->| Ende  |
  +-------+     +-------+
  ```
  ````

### Matrix-Quiz

- **Beschreibung:** Komplexe Quiz-Matrizen mit mehreren Zeilen
- **Regex-Muster:** `(\[\[[^\]]+\]\]\s*)+\n(\[\[[^\]]+\]\]\s*)+`
- **Beispiel:**
  ```markdown
  [[Berlin][Paris][London]]
  [[Deutschland][Frankreich][England]]
  ```

### Galerien

- **Beschreibung:** Mehrere Bilder in einer Zeile/Absatz
- **Regex-Muster:** `!\[.*?\]\(.*?\)\s*!\[.*?\]\(.*?\)`
- **Beispiel:**
  ```markdown
  ![Bild1](img1.png) ![Bild2](img2.png) ![Bild3](img3.png)
  ```

### Strukturelle Analyse

- **Beschreibung:** Überschriften für Kursstruktur
- **H1 Regex:** `^#\s+[^#]`
- **H2 Regex:** `^##\s+[^#]`
- **H3 Regex:** `^###\s+[^#]`

### Links

- **Beschreibung:** Standard Markdown-Links
- **Regex-Muster:** `\[([^\]]+)\]\(([^\)]+)\)`

### Kommentare im Body

- **Beschreibung:** HTML-Kommentare außerhalb des Headers
- **Erkennung:** Alle `<!-- ... -->` minus Header-Kommentar

---

## Template-URL-Normalisierung

Template-URLs werden normalisiert, um Duplikate zu vermeiden:

1. Lowercase-Konvertierung
2. Entfernung von `/refs/heads/`
3. Konvertierung von `github.com/.../blob/` zu `raw.githubusercontent.com/`

**Beispiel:**
- `https://raw.githubusercontent.com/LiaTemplates/Tikz-Jax/main/README.md`
- `https://raw.githubusercontent.com/liatemplates/tikz-jax/main/readme.md`
→ Werden als gleich behandelt

---

*Generiert aus `analyzeLiaScriptFeatures.py` - Aktualisiert: 2026-01-19*
