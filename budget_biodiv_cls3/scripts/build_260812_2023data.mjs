import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const documentDir = "C:\\repos\\biofin-research\\budget_biodiv_cls3\\document";
const targetPath = path.join(documentDir, "BIOFIN_2023_통계차트 및 취합데이터.csv");
const primaryPath = path.join(documentDir, "2023biofin_label_matched.csv");
const fallbackPath = path.join(documentDir, "BIOFIN_전체취합_사업설명자료매칭.csv");
const outputPath = path.join(documentDir, "260812_2023data.csv");
const previewPath = path.join(documentDir, "260812_2023data_preview.png");

const decode = async (filePath, encoding) => {
  const bytes = await fs.readFile(filePath);
  return new TextDecoder(encoding).decode(bytes);
};

const sheetValues = async (csvText, sheetName) => {
  const workbook = await Workbook.fromCSV(csvText, { sheetName });
  return workbook.worksheets.getItem(sheetName).getUsedRange(true).values;
};

const clean = (value) => String(value ?? "").trim();
const normalize = (value) => clean(value).toLowerCase().replace(/[^0-9a-z가-힣]/g, "");
const hierarchyKey = (row) => row.slice(1, 10).map(normalize).join("|");

const targetRows = await sheetValues(await decode(targetPath, "euc-kr"), "Target");
const primaryRows = await sheetValues(await decode(primaryPath, "utf-8"), "Primary");
const fallbackRows = await sheetValues(await decode(fallbackPath, "utf-8"), "Fallback");

const targetHeader = targetRows[0].map(clean);
const primaryHeader = primaryRows[0].map(clean);
const fallbackHeader = fallbackRows[0].map(clean);
const targetData = targetRows.slice(1);
const primaryData = primaryRows.slice(1);
const fallbackData = fallbackRows.slice(1);

const byNo = (rows) => new Map(rows.map((row) => [clean(row[0]), row]));
const primaryByNo = byNo(primaryData);
const fallbackByNo = byNo(fallbackData);

const documentColumns = [
  "business_key",
  "사업설명자료_파일명",
  "사업설명자료_상대경로",
  "사업설명자료_절대경로",
  "문서매칭상태",
  "문서매칭방식",
  "문서매칭후보수",
];

const primaryDocIndexes = documentColumns.map((name) => primaryHeader.indexOf(name));
const fallbackDocIndexes = documentColumns.map((name) => fallbackHeader.indexOf(name));
if (primaryDocIndexes.some((index) => index < 0) || fallbackDocIndexes.some((index) => index < 0)) {
  throw new Error("참고 CSV에 필요한 문서 매칭 열이 없습니다.");
}

const outputHeader = [...targetHeader];
outputHeader[15] = "BIOFIN 1차 카테고리";
outputHeader[19] = "하위 카테고리";
outputHeader.push(...documentColumns, "문서매칭출처");

const stats = {
  targetRows: targetData.length,
  primaryMatches: 0,
  fallbackMatches: 0,
  noDocument: 0,
  hierarchyConflicts: 0,
  categoryZerosFilled: { BIOFIN분류: 0, "BIOFIN 1차 카테고리": 0, 하위: 0, "하위 카테고리": 0 },
};

const outputData = targetData.map((sourceRow) => {
  const row = [...sourceRow];
  for (const [index, label] of [[14, "BIOFIN분류"], [15, "BIOFIN 1차 카테고리"], [16, "하위"], [19, "하위 카테고리"]]) {
    if (!clean(row[index])) {
      row[index] = "0";
      stats.categoryZerosFilled[label] += 1;
    }
  }

  const no = clean(row[0]);
  const primary = primaryByNo.get(no);
  const fallback = fallbackByNo.get(no);
  let reference = null;
  let referenceIndexes = null;
  let source = "";

  if (primary && hierarchyKey(primary) === hierarchyKey(sourceRow)) {
    reference = primary;
    referenceIndexes = primaryDocIndexes;
    source = "2023biofin_label_matched.csv";
    stats.primaryMatches += 1;
  } else if (fallback && hierarchyKey(fallback) === hierarchyKey(sourceRow)) {
    reference = fallback;
    referenceIndexes = fallbackDocIndexes;
    source = "BIOFIN_전체취합_사업설명자료매칭.csv";
    stats.fallbackMatches += 1;
  } else {
    stats.hierarchyConflicts += 1;
  }

  const documentValues = reference
    ? referenceIndexes.map((index) => clean(reference[index]))
    : ["", "", "", "", "NO_DOCUMENT", "NO_REFERENCE_MATCH", "0"];
  if (documentValues[4] !== "MATCHED") stats.noDocument += 1;
  return [...row, ...documentValues, source];
});

const csvEscape = (value) => {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const csvText = [outputHeader, ...outputData]
  .map((row) => row.map(csvEscape).join(","))
  .join("\r\n");
await fs.writeFile(outputPath, `\uFEFF${csvText}`, "utf8");

// 최종 CSV를 다시 가져와 값과 시각적 가독성을 검증한다.
const verificationWorkbook = await Workbook.fromCSV(`\uFEFF${csvText}`, { sheetName: "2023data" });
const verificationSheet = verificationWorkbook.worksheets.getItem("2023data");
verificationSheet.freezePanes.freezeRows(1);
verificationSheet.getRangeByIndexes(0, 0, 1, outputHeader.length).format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
verificationSheet.getRange("A1:H20").format.autofitColumns();
const preview = await verificationWorkbook.render({
  sheetName: "2023data",
  range: "A1:H20",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const inspection = await verificationWorkbook.inspect({
  kind: "table",
  range: `2023data!A1:AC6`,
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 29,
  maxChars: 5000,
});
const errors = await verificationWorkbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "final formula error scan",
});

console.log(JSON.stringify({ outputPath, previewPath, columns: outputHeader.length, ...stats }, null, 2));
console.log(inspection.ndjson);
console.log(errors.ndjson);
