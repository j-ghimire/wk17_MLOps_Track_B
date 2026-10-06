import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Article:
    title: str
    category: str
    content: str
    terms: frozenset[str]


_ARTICLES = [
    ("Password recovery", "ACCOUNT", "Use the password reset link on the sign-in page. If the reset email does not arrive, check spam and confirm the account email.", "password reset login sign in account email"),
    ("Card payment declined", "CARD", "Check that the card is active, has sufficient funds, and that online payments are enabled. A pending authorization can also affect available balance.", "card payment declined rejected cash balance"),
    ("Cash withdrawal issue", "CASH_WITHDRAWAL", "Check the ATM network, daily withdrawal limit, and available balance. Never share your PIN with support.", "cash withdrawal atm pin limit balance"),
    ("Cash deposit issue", "CASH_DEPOSIT", "Cash deposits can take time to appear. Keep the receipt and contact support if the balance is not updated after the stated processing window.", "cash deposit balance receipt processing"),
    ("Transfer pending", "TRANSFER", "A transfer may remain pending while security or recipient checks complete. Confirm the recipient details and do not submit duplicate transfers.", "transfer pending recipient bank duplicate"),
    ("Refund status", "REFUND", "Refund timing depends on the merchant and payment network. Keep the merchant confirmation and check the original payment method.", "refund merchant returned money payment"),
    ("Exchange rate", "EXCHANGE_RATE", "The applied exchange rate can include the network rate and a disclosed service fee. Review the transaction receipt for the exact breakdown.", "exchange rate currency fee receipt"),
    ("Card delivery", "CARD_DELIVERY", "Track card delivery from the Cards section. Verify the delivery address and contact support if tracking has not changed.", "card delivery shipping track address"),
    ("Card stolen", "CARD", "Freeze a lost or stolen card immediately in the app, then contact support through the verified channel.", "stolen lost freeze card support"),
    ("Top up", "TOP_UP", "Use an eligible payment method in the top-up flow. Check limits and confirm the source card supports the transaction.", "top up add money recharge payment"),
    ("ATM fee", "CASH_WITHDRAWAL", "ATM operators and account plans may charge fees. Review the fee disclosure and use an in-network ATM when possible.", "atm fee charge in network"),
]


def _terms(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


ARTICLES = [Article(title, category, content, _terms(f"{title} {terms}")) for title, category, content, terms in _ARTICLES]


def search_articles(query: str, limit: int = 3) -> list[tuple[Article, float]]:
    if not query or not query.strip():
        return []
    query_terms = _terms(query)
    scored: list[tuple[Article, float]] = []
    for article in ARTICLES:
        overlap = len(query_terms & article.terms)
        score = overlap / max(len(query_terms), 1)
        if score > 0:
            scored.append((article, round(score, 3)))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]
