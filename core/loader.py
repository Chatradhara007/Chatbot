import pymongo
import config
from langchain_core.documents import Document


def load_and_chunk_data():
    """
    Load from MongoDB — company_details, company_records, branch_reports, AND batch_reports.
    Builds one Document per company, one per branch, one per batch.
    """
    client = pymongo.MongoClient(config.MONGO_URI)
    db = client["placementsDB"]

    documents = []

    # ── 1. Company documents (company_details + company_records) ─────────────
    companies = list(db["company_details"].find({}))

    for company in companies:
        cid    = company.get("companyID", "")
        name   = company.get("companyName", cid)
        sector = company.get("companySector", "N/A")
        desc   = company.get("companyDesc", "N/A")

        content = (
            f"Company: {name}\n"
            f"Sector: {sector}\n"
            f"About: {desc}\n"
        )

        records = list(db["company_records"].find({"companyID": cid}, {"_id": 0}))
        for record in records:
            year   = record.get("year", "Unknown Year")
            offers = record.get("totalOffers", 0)

            role_parts = []
            for r in record.get("roles", []):
                r_type = r.get("roleType", "Unknown Role")
                r_ctc  = r.get("roleCTC", "N/A")
                role_parts.append(f"{r_type} (Package: {r_ctc})")

            roles_str = ", ".join(role_parts) if role_parts else "No role details available"
            content += f"In {year}, {name} made {offers} total offer(s): {roles_str}.\n"

        doc = Document(page_content=content, metadata={"type": "company", "company": name})
        documents.append(doc)

    # ── 2. Branch documents (branch_reports) ─────────────────────────────────
    branches = list(db["branch_reports"].find({}))

    for branch in branches:
        b_name   = branch.get("branch", str(branch.get("_id", "Unknown")))
        on_rolls = branch.get("onRolls", "N/A")
        reg      = branch.get("registered", "N/A")
        eligible = branch.get("eligible", "N/A")
        not_elig = branch.get("notEligible", "N/A")
        placed   = branch.get("placed", "N/A")
        multiple = branch.get("multiple", "N/A")
        no_multi = branch.get("noOfStudentsMultiple", "N/A")
        t_offers = branch.get("totalOffers", "N/A")
        pct      = branch.get("placementPercent", "N/A")
        unplaced = branch.get("unplaced", "N/A")
        highest  = branch.get("highestSalary", "N/A")
        lowest   = branch.get("lowestSalary", "N/A")
        average  = branch.get("averageSalary", "N/A")

        content = (
            f"Branch: {b_name}\n"
            f"Students on rolls: {on_rolls}\n"
            f"Registered for placements: {reg}\n"
            f"Eligible students: {eligible}\n"
            f"Not eligible students: {not_elig}\n"
            f"Students placed: {placed}\n"
            f"Students unplaced: {unplaced}\n"
            f"Students with multiple offers: {no_multi}\n"
            f"Total offers received: {t_offers}\n"
            f"Instances of multiple companies selecting same student: {multiple}\n"
            f"Placement percentage: {pct}%\n"
            f"Highest salary offered: {highest} LPA\n"
            f"Lowest salary offered: {lowest} LPA\n"
            f"Average salary offered: {average} LPA\n"
        )

        doc = Document(
            page_content=content,
            metadata={"type": "branch", "branch": b_name}
        )
        documents.append(doc)

    # ── 3. Batch documents (batch_reports) ────────────────────────────────────
    batches = list(db["batch_reports"].find({}))

    for batch in batches:
        b_name   = batch.get("batch", "Unknown")
        t_stu    = batch.get("total_students", "N/A")
        placed   = batch.get("placed", "N/A")
        pct      = batch.get("placement_percentage", "N/A")
        avg_pkg  = batch.get("avg_package", "N/A")
        high_pkg = batch.get("highest_package", "N/A")

        content = (
            f"Batch: {b_name}\n"
            f"Total students in batch: {t_stu}\n"
            f"Total students placed: {placed}\n"
            f"Overall placement percentage: {pct}%\n"
            f"Average package: {avg_pkg} LPA\n"
            f"Highest package: {high_pkg} LPA\n"
        )

        doc = Document(
            page_content=content,
            metadata={"type": "batch", "batch": b_name}
        )
        documents.append(doc)

    print(f"[Loader] Built {len(documents)} documents "
          f"({len(companies)} companies, {len(branches)} branches, {len(batches)} batches).")
    return documents
