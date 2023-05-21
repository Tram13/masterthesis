% CLASS file for bwthesis
% Based on the book class from Latex
% author: Joris Meys
% 2017/05/29

% --- Class structure identification

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bwthesis}[2017/05/29 v1.1 bwthesis]
% --- Class structure declaration of options
%
%  This class extends book class

% Color definitions for titles. The actual colors are defined
% in bwtitlepage
\newcommand{\coltitle}{\color{black}}
\newcommand{\coltitlepage}{\color{UGBlue}}

% Add new option
\DeclareOption{usecolor}{
	\renewcommand{\coltitle}{\color{UGBlue}}
}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}

% --- Class structure execution of options
%
\ProcessOptions \relax

\LoadClass{book}

% --- Loading necessary packages
\RequirePackage[margin = 2.5cm]{geometry}
\RequirePackage{bwtitlepage} % To construct the title page
\RequirePackage{caption} % To adjust captions

% --- adjust binding offset
%\if@twoside\geometry{bindingoffset = 1cm}\fi

% --- adjust captions
\captionsetup[table]{skip=1.5ex}

% --- Define the standard lettertype: Arev

\RequirePackage{arev}

% --- Set up the layout parameters

% no indentations
\setlength{\parindent}{0cm}
% more space between paragraphs
\setlength{\parskip}{2.3ex plus 0.3ex minus 0.3ex}
% makes all pages the height of the text on that page, and no extra vertical space is added
\raggedbottom

% define tocdepth
\setcounter{tocdepth}{2}
% set lenght of headheight
\setlength{\headheight}{21pt}
% to suppress widows and orphans
\widowpenalty=10000
\clubpenalty=10000

% --- Set up the headers and footers
\RequirePackage{fancyhdr}

% for the frontmatter
\if@twoside
  \fancypagestyle{frontmatter}{
    \fancyhf{}
    \renewcommand{\headrulewidth}{0pt}
    \fancyfoot[RO]{}
    \fancyfoot[LE]{}
    }
\else
  \fancypagestyle{frontmatter}{
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
	\fancyfoot[RO]{}
  }
\fi

% for the mainmatter
\renewcommand{\chaptermark}[1]{\markboth{\thechapter. \ #1}{}}

\if@twoside
  \fancypagestyle{mainmatter}{
    \fancyhf{}
    \fancyhead[LO]{ \leftmark}
  %	\fancyhead[RE]{ \rightmark}
    \fancyfoot[RO]{\thepage}
    \fancyfoot[LE]{\thepage}
    \renewcommand{\headrulewidth}{0.15pt}
  }
\else
  \fancypagestyle{mainmatter}{
    \fancyhf{}
    \fancyhead[LO]{ \leftmark}
    \fancyfoot[RO]{\thepage}
    \renewcommand{\headrulewidth}{0.15pt}
  }
\fi

% for the title pages
\if@twoside
  \fancypagestyle{plain}{
    \fancyhf{}
    \fancyhead[LO]{}
  	\fancyhead[RE]{}
    \fancyfoot[RO]{\thepage}
  	\fancyfoot[LE]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
  }
\else
  \fancypagestyle{plain}{
    \fancyhf{}
    \fancyhead[LO]{}
    \fancyfoot[RO]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
  }
\fi
% --- Set up the front matter
\renewcommand{\listfigurename}{Lijst van Figuren}
\renewcommand{\listtablename}{Lijst van Tabellen}

% --- Set up Chapter titles etc
\RequirePackage[raggedright, explicit]{titlesec}

\titleformat{\chapter} % command -> in dit stukje aanpassingen gemaakt ivm "Hoofdstuk:"
  [block] % shape
  {\Huge \selectfont \coltitle \bf } % format
  {\thechapter.} % the label
  { 1ex } %separator space
  {\MakeUppercase{\uline{#1}} } %before code
  { } %after code

% Change the spacing above the chapter. This is left - above - below
\titlespacing*{\chapter}{0pt}{0pt}{40pt}[15pt]

% --- Setup the document

\AtBeginDocument{
  \maketitle
}
