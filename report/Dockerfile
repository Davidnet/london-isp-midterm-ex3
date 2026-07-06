FROM pandoc/latex:latest-ubuntu
RUN tlmgr update --self \
  && tlmgr install \
    tufte-latex hardwrap titlesec xifthen ifmtarg changepage paralist \
    placeins ragged2e sauerj setspace textcase fmtcount catchfile \
    transparent morefloats xkeyval xltxtra realscripts metalogo mparhack
