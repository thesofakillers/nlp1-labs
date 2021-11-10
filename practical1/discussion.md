# Discussion

> Based on your experiments, what are the **effective features and techniques**
> in sentiment analysis? What **information** do different features encode? Why
> is this important? What are the **limitations** of these features and
> techniques?

Based on our experiments, some effective features in Sentiment Analysis are
token occurrence and parts of speech (POS) tags. The former acts as a weight
associating a given token to a sentiment. POS tags add dimensionality to token
occurrence, helping models to disambiguate if possible. This is important given
the (lexical, semantic and syntactic) ambiguity that pervades languages. This is
limited though in that often POS is not enough to disambiguate sentiment:
consider the word "match" for instance, which can have several semantic meanings
for the same POS form or the word "and" whose neutrality is unrelated to its
POS. The latter case can be dealt with by not considering closed-class POS tags,
as was done in section 3.3, while the former is simply an intrinsic limitation
of this feature.

Some effective techniques that use these features are Naive Bayes (NB) and
Support Vector Machines (SVM's). We do not consider lexicon-based approaches
effective given their lower performance and reliance on a provided lexicon. A
classic limitation of NB is the fact that context is lost given our independence
assumption. What this means is that word order also does not affect our
predictions. This may be problematic for sentiment analysis given long-distance
dependencies often present in language. Issues with the independence assumption
can be noticed by considering the phrases "bad not good" and "good not bad"
which are permutations of the same set of words resulting in different
semantics. This issue can be somewhat addressed by using n-grams rather than
unigrams. This however has memory implications, with vocabulary sizes greatly
increasing with the order of the n-grams used as discussed in 2.10. While this
limitation is intrinsic to NB models, it is not necessarily present in SVM's.
However, since in our approach we were simply using token occurrence (optionally
augmented with POS tags) as features, we were implicitly incorporating this
assumption into our SVM training, making little to no use of context.

While throughout the work we utilized accuracy to evaluate our models, as
discussed in 2.2 this is a limited metric which could be replaced by other
metrics more suitable to the problem. For instance, from Fig. 2, when using
accuracy, the trend appears to suggest that SVM's outperform NB models. However,
the precision and recall metrics for these models seem to suggest that NB
trained on n-gram combinations are better suited for problems sensitive to
precision, while SVMs are better suited for problems sensitive to recall. That
being said, SVM's were not trained on n-grams and a comparison may therefore not
be suitable. Future work could consider training SVM's with n-gram features, or
exploring models with better context management.
