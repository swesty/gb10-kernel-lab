cd /usr/local/lib/python3.12/dist-packages
F=b12x/attention/paged/_selected_forward.py
echo "### top constants"; grep -nE "^_[A-Z_]+ *=|^def |^class |^@" $F | head -40
echo "### launcher functions"; sed -n 1330,1608p $F
