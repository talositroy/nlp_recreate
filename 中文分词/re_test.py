# 这是一个关于正则表达式的练习
import re

text = r"Formally, a regular expression is an algebraic notation for characterizing a set of \
strings. Thustheycanbeusedtospecifysearchstringsaswellastodefinealanguagein \
aformalway. We willbeginbytalkingaboutregularexpressionsasa wayofspecifying \
searches in texts, and proceed to other uses. Section 2.3 shows that the use of just \
three regular expression operators is sufficient to characterize strings, but we use the \
more convenient and commonly-used regular expression syntax of the Perl language \
throughout this section. Since common text-processing programs agree on most of the \
syntax of regular expressions, most of what we say extends to all UNIX, Microsoft \
Word, and WordPerfect regular expressions. Appendix A shows the few areas where \
these programs differ from the Perl syntax."

p_string = text.split('\\')

# for i in range(len(p_string)):
#     if (re.search('\s.\..\s', p_string[i])):
#         print(i, ':::', p_string[i])


for i in range(len(p_string)):
    if (re.search('^.', p_string[i])):
        print(i, ':::', p_string[i])