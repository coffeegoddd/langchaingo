package mariadb

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"database/sql"
	"encoding/json"

	_ "github.com/go-sql-driver/mysql"

	"github.com/google/uuid"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

var (
	ErrEmbedderWrongNumberVectors = errors.New("number of vectors from embedder does not match number of documents")
	ErrInvalidScoreThreshold      = errors.New("score threshold must be between 0 and 1")
	ErrInvalidFilters             = errors.New("invalid filters")
	ErrUnsupportedOptions         = errors.New("unsupported options")
)

// DB represents both a sql.DB and sql.Tx
type DB interface {
	PingContext(ctx context.Context) error
	BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error)
	ExecContext(ctx context.Context, sql string, arguments ...any) (sql.Result, error)
	QueryContext(ctx context.Context, sql string, arguments ...any) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, sql string, arguments ...any) *sql.Row
}

type CloseNoErr interface {
	Close()
}

// Store is a wrapper around the mariadb client.
type Store struct {
	embedder                              embeddings.Embedder
	connURL                               string
	db                                    DB
	embeddingTableName                    string
	collectionTableName                   string
	databaseName                          string
	databaseUUID                          string
	databaseMetadata                      map[string]any
	preDeleteDatabase                     bool
	vectorDimensions                      int
	createEmbeddingIndexAfterAddDocuments bool
}

var _ vectorstores.VectorStore = Store{}

// New creates a new Store with options.
func New(ctx context.Context, opts ...Option) (Store, error) {
	store, err := applyClientOptions(opts...)
	if err != nil {
		return Store{}, err
	}
	if store.db == nil {
		store.db, err = sql.Open("mysql", store.connURL)
		if err != nil {
			return Store{}, err
		}
	}
	if err = store.db.PingContext(ctx); err != nil {
		return Store{}, err
	}
	if err = store.init(ctx); err != nil {
		return Store{}, err
	}
	return store, nil
}

// Close closes the db.
func (s Store) Close() error {
	if closer, ok := s.db.(io.Closer); ok {
		return closer.Close()
	}
	if closer, ok := s.db.(CloseNoErr); ok {
		closer.Close()
	}
	return nil
}

func (s *Store) init(ctx context.Context) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}

	if err := s.createCollectionTableIfNotExists(ctx, tx); err != nil {
		return err
	}
	if err := s.createEmbeddingTableIfNotExists(ctx, tx); err != nil {
		return err
	}

	if s.preDeleteDatabase {
		if err := s.RemoveDatabase(ctx, tx); err != nil {
			return err
		}
	}
	if err := s.createOrGetDatabase(ctx, tx); err != nil {
		return err
	}

	return tx.Commit()
}

func (s Store) createCollectionTableIfNotExists(ctx context.Context, tx *sql.Tx) error {
	sql := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
	name varchar(720),
	cmetadata json,
	`+"`uuid`"+` varchar(36) NOT NULL,
	UNIQUE (name),
	PRIMARY KEY (uuid))`, s.collectionTableName)
	if _, err := tx.ExecContext(ctx, sql); err != nil {
		return err
	}
	return nil
}

func (s Store) createEmbeddingTableIfNotExists(ctx context.Context, tx *sql.Tx) error {
	embeddingStr := "embedding VECTOR NOT NULL,"
	if s.vectorDimensions != 0 {
		embeddingStr = fmt.Sprintf("embedding VECTOR(%d) NOT NULL,", s.vectorDimensions)
	}
	sql := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
	collection_id varchar(36),
	`+embeddingStr+`
	document longtext,
	cmetadata json,
	`+"`uuid`"+` varchar(36) NOT NULL,
	CONSTRAINT langchain_mariadb_embedding_collection_id_fkey
	FOREIGN KEY (collection_id) REFERENCES %s (uuid) ON DELETE CASCADE,
	PRIMARY KEY (uuid))`, s.embeddingTableName, s.collectionTableName)
	if _, err := tx.ExecContext(ctx, sql); err != nil {
		return err
	}

	sql = fmt.Sprintf(`SET @index_name = '%s_collection_id';
SET @table_name = '%s';

SELECT COUNT(*)
INTO @index_exists
FROM information_schema.statistics
WHERE table_schema = DATABASE()
  AND table_name = @table_name
  AND index_name = @index_name;

SET @sql = IF(@index_exists = 0, CONCAT('CREATE INDEX ', @index_name, ' ON ', @table_name, ' (collection_id)'), 'SELECT ''Index already exists''');

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;`, s.embeddingTableName, s.embeddingTableName)
	if _, err := tx.ExecContext(ctx, sql); err != nil {
		return err
	}

	if !s.createEmbeddingIndexAfterAddDocuments {
		sql = fmt.Sprintf(`SET @index_name = '%s_embedding_idx';
	SET @table_name = '%s';
	
	SELECT COUNT(*)
	INTO @index_exists
	FROM information_schema.statistics
	WHERE table_schema = DATABASE()
		AND table_name = @table_name
		AND index_name = @index_name;
	
	SET @sql = IF(@index_exists = 0, CONCAT('CREATE VECTOR INDEX M=8 DISTANCE=cosine', @index_name, ' ON ', @table_name, ' (embedding)'), 'SELECT ''Index already exists''');
	
	PREPARE stmt FROM @sql;
	EXECUTE stmt;
	DEALLOCATE PREPARE stmt;`, s.embeddingTableName, s.embeddingTableName)
		if _, err := tx.ExecContext(ctx, sql); err != nil {
			return err
		}
	}

	return nil
}

// AddDocuments adds documents to the MariaDB database associated with 'Store'.
// and returns the ids of the added documents.
func (s Store) AddDocuments(
	ctx context.Context,
	docs []schema.Document,
	options ...vectorstores.Option,
) ([]string, error) {
	opts := s.getOptions(options...)
	if opts.ScoreThreshold != 0 || opts.Filters != nil || opts.NameSpace != "" {
		return nil, ErrUnsupportedOptions
	}

	docs = s.deduplicate(ctx, opts, docs)

	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		texts = append(texts, doc.PageContent)
	}

	embedder := s.embedder
	if opts.Embedder != nil {
		embedder = opts.Embedder
	}
	vectors, err := embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return nil, err
	}

	if len(vectors) != len(docs) {
		return nil, ErrEmbedderWrongNumberVectors
	}

	ids := make([]string, len(docs))
	valueStrings := make([]string, 0, len(docs))
	valueArgs := make([]interface{}, 0, len(docs)*2)
	for docIdx, doc := range docs {
		id := uuid.New().String()
		ids[docIdx] = id
		valueStrings = append(valueStrings, "(?, ?, ?, ?, ?)")
		jsonEmbedding, err := json.Marshal(vectors[docIdx])
		if err != nil {
			return nil, err
		}
		jsonMetadata, err := json.Marshal(doc.Metadata)
		if err != nil {
			return nil, err
		}
		valueArgs = append(valueArgs, id, doc.PageContent, jsonEmbedding, jsonMetadata, s.databaseUUID)
	}

	sql := fmt.Sprintf(`INSERT INTO %s (`+"`uuid`"+`, document, embedding, cmetadata, collection_id)
	VALUES %s`, s.embeddingTableName, strings.Join(valueStrings, ","))

	_, err = s.db.ExecContext(ctx, sql, valueArgs...)
	if err != nil {
		return nil, err
	}

	if s.createEmbeddingIndexAfterAddDocuments {
		sql := fmt.Sprintf(`SET @index_name = '%s_embedding_idx';
	SET @table_name = '%s';

	SELECT COUNT(*)
	INTO @index_exists
	FROM information_schema.statistics
	WHERE table_schema = DATABASE()
		AND table_name = @table_name
		AND index_name = @index_name;

	SET @sql = IF(@index_exists = 0, CONCAT('CREATE VECTOR INDEX M=8 DISTANCE=cosine', @index_name, ' ON ', @table_name, ' (embedding)'), 'SELECT ''Index already exists''');

	PREPARE stmt FROM @sql;
	EXECUTE stmt;
	DEALLOCATE PREPARE stmt;`, s.embeddingTableName, s.embeddingTableName)
		if _, err := s.db.ExecContext(ctx, sql); err != nil {
			return nil, err
		}
	}

	return ids, nil
}

//nolint:cyclop
func (s Store) SimilaritySearch(
	ctx context.Context,
	query string,
	numDocuments int,
	options ...vectorstores.Option,
) ([]schema.Document, error) {
	opts := s.getOptions(options...)
	databaseName := s.getDatabaseName(opts)
	scoreThreshold, err := s.getScoreThreshold(opts)
	if err != nil {
		return nil, err
	}
	filter, err := s.getFilters(opts)
	if err != nil {
		return nil, err
	}
	embedder := s.embedder
	if opts.Embedder != nil {
		embedder = opts.Embedder
	}
	embedderData, err := embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	whereQuerys := make([]string, 0)
	if scoreThreshold != 0 {
		whereQuerys = append(whereQuerys, fmt.Sprintf("data.distance < %f", 1-scoreThreshold))
	}
	for k, v := range filter {
		whereQuerys = append(whereQuerys, fmt.Sprintf("(data.cmetadata ->> '%s') = '%s'", k, v))
	}
	whereQuery := strings.Join(whereQuerys, " AND ")
	if len(whereQuery) == 0 {
		whereQuery = "TRUE"
	}

	jsonEmbedding, err := json.Marshal(embedderData)
	if err != nil {
		return nil, err
	}

	sql := fmt.Sprintf(`WITH filtered_embedding_dims AS (
    SELECT
        *
    FROM
        %s
)
SELECT
    data.document,
    data.cmetadata,
    (1 - data.distance) AS score
FROM
(
    SELECT
        f.*,
        VEC_DISTANCE_COSINE(f.embedding, Vec_FromText(?)) AS distance
    FROM
        filtered_embedding_dims AS f
        JOIN %s AS t ON f.collection_id = t.uuid
    WHERE
        t.name = '%s'
) AS data
WHERE %s
ORDER BY
    data.distance
LIMIT
    ?;`, s.embeddingTableName,
		s.collectionTableName,
		databaseName,
		whereQuery)

	// todo: remove this
	// err = s.printExplainData(ctx, databaseName, jsonEmbedding)
	// if err != nil {
	// 	return nil, err
	// }

	// err = s.printEmbeddingRowsOfSameLength(ctx)
	// if err != nil {
	// 	return nil, err
	// }

	err = s.printData(ctx, databaseName, jsonEmbedding)
	if err != nil {
		return nil, err
	}

	err = s.printDataWithOrderBy(ctx, databaseName, whereQuery, jsonEmbedding)
	if err != nil {
		return nil, err
	}

	rows, err := s.db.QueryContext(ctx, sql, jsonEmbedding, numDocuments)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	docs := make([]schema.Document, 0)
	for rows.Next() {
		var content string
		var metadata string
		var score float64

		if err := rows.Scan(&content, &metadata, &score); err != nil {
			return nil, err
		}

		var metadataMap map[string]any
		if err := json.Unmarshal([]byte(metadata), &metadataMap); err != nil {
			return nil, err
		}

		docs = append(docs, schema.Document{
			PageContent: content,
			Metadata:    metadataMap,
			Score:       float32(score),
		})
	}
	return docs, rows.Err()
}

func (s *Store) printExplainData(ctx context.Context, databaseName string, jsonEmbedding []byte) error {
	sql := fmt.Sprintf(`EXPLAIN WITH filtered_embedding_dims AS (
    SELECT
        *
    FROM
        %s
)
SELECT
    data.document,
    data.cmetadata,
    (1 - data.distance) AS score
FROM
(
    SELECT
        f.*,
        VEC_DISTANCE_COSINE(f.embedding, Vec_FromText(?)) AS distance
    FROM
        filtered_embedding_dims AS f
        JOIN %s AS t ON f.collection_id = t.uuid
    WHERE
        t.name = '%s'
) AS data`, s.embeddingTableName, s.collectionTableName, databaseName)

	var id string
	var selectType string
	var table string
	var tType string
	var possibleKeys string
	var key string
	var keyLen string
	var ref string
	var rows string
	var extra string

	row := s.db.QueryRowContext(ctx, sql, jsonEmbedding)
	err := row.Scan(&id, &selectType, &table, &tType, &possibleKeys, &key, &keyLen, &ref, &rows, &extra)
	if err != nil {
		return err
	}

	fmt.Printf("DUSTIN: mariadb: explain select: id: %s\n", id)
	fmt.Printf("DUSTIN: mariadb: explain select: select_type: %s\n", selectType)
	fmt.Printf("DUSTIN: mariadb: explain select: table: %s\n", table)
	fmt.Printf("DUSTIN: mariadb: explain select: type: %s\n", tType)
	fmt.Printf("DUSTIN: mariadb: explain select: possible_keys: %s\n", possibleKeys)
	fmt.Printf("DUSTIN: mariadb: explain select: key: %s\n", key)
	fmt.Printf("DUSTIN: mariadb: explain select: key_len: %s\n", keyLen)
	fmt.Printf("DUSTIN: mariadb: explain select: ref: %s\n", ref)
	fmt.Printf("DUSTIN: mariadb: explain select: rows: %s\n", rows)
	fmt.Printf("DUSTIN: mariadb: explain select: Extra: %s\n", extra)

	return nil
}

func (s *Store) printEmbeddingRowsOfSameLength(ctx context.Context) error {
	sql := fmt.Sprintf("SELECT * FROM %s;", s.embeddingTableName)
	rows, err := s.db.QueryContext(ctx, sql)
	if err != nil {
		return err
	}
	defer rows.Close()

	//docs := make([]schema.Document, 0)
	for rows.Next() {
		var collectionID string
		var embedding string
		var document string
		var cmetadata string
		var uuidStr string
		if err := rows.Scan(&collectionID, &embedding, &document, &cmetadata, &uuidStr); err != nil {
			return err
		}

		fmt.Printf("DUSTIN: printEmbeddingRowsOfSameLength: %s, %s, %s\n", uuidStr, collectionID, document)
	}
	return rows.Err()
}

func (s *Store) printData(ctx context.Context, databaseName string, jsonEmbedding []byte) error {
	sql := fmt.Sprintf(`WITH filtered_embedding_dims AS (
    SELECT
        *
    FROM
        %s
)
SELECT
    data.document,
    data.cmetadata,
    (1 - data.distance) AS score
FROM
(
    SELECT
        f.*,
        VEC_DISTANCE_COSINE(f.embedding, Vec_FromText(?)) AS distance
    FROM
        filtered_embedding_dims AS f
        JOIN %s AS t ON f.collection_id = t.uuid
    WHERE
        t.name = '%s'
) AS data`, s.embeddingTableName, s.collectionTableName, databaseName)

	rows, err := s.db.QueryContext(ctx, sql, jsonEmbedding)
	if err != nil {
		return err
	}
	defer rows.Close()

	//docs := make([]schema.Document, 0)
	for rows.Next() {
		//var collectionID string
		//var embedding string
		var document string
		var cmetadata string
		var score float64
		//var uuidStr string
		if err := rows.Scan(&document, &cmetadata, &score); err != nil {
			return err
		}

		fmt.Printf("DUSTIN: printData: %s, %s, %f\n", document, cmetadata, score)
	}
	return rows.Err()
}

func (s *Store) printDataWithOrderBy(ctx context.Context, databaseName, whereQuery string, jsonEmbedding []byte) error {
	sql := fmt.Sprintf(`WITH filtered_embedding_dims AS (
    SELECT
        *
    FROM
        %s
)
SELECT
    data.document,
    data.cmetadata,
    (1 - data.distance) AS score
FROM
(
    SELECT
        f.*,
        VEC_DISTANCE_COSINE(f.embedding, Vec_FromText(?)) AS distance
    FROM
        filtered_embedding_dims AS f
        JOIN %s AS t ON f.collection_id = t.uuid
    WHERE
        t.name = '%s'
) AS data WHERE %s
ORDER BY
    data.distance`, s.embeddingTableName, s.collectionTableName, databaseName, whereQuery)

	rows, err := s.db.QueryContext(ctx, sql, jsonEmbedding)
	if err != nil {
		return err
	}
	defer rows.Close()

	//docs := make([]schema.Document, 0)
	for rows.Next() {
		//var collectionID string
		//var embedding string
		var document string
		var cmetadata string
		var score float64
		//var uuidStr string
		if err := rows.Scan(&document, &cmetadata, &score); err != nil {
			return err
		}

		fmt.Printf("DUSTIN: printDataWithOrderBy: %s, %s, %f\n", document, cmetadata, score)
	}
	return rows.Err()
}

//nolint:cyclop
func (s Store) Search(
	ctx context.Context,
	numDocuments int,
	options ...vectorstores.Option,
) ([]schema.Document, error) {
	opts := s.getOptions(options...)
	databaseName := s.getDatabaseName(opts)
	filter, err := s.getFilters(opts)
	if err != nil {
		return nil, err
	}
	whereQuerys := make([]string, 0)
	for k, v := range filter {
		whereQuerys = append(whereQuerys, fmt.Sprintf("(%s.cmetadata ->> '%s') = '%s'", s.embeddingTableName, k, v))
	}
	whereQuery := strings.Join(whereQuerys, " AND ")
	if len(whereQuery) == 0 {
		whereQuery = "TRUE"
	}
	sql := fmt.Sprintf(`SELECT
	%s.document,
	%s.cmetadata
FROM %s
JOIN %s ON %s.collection_id=%s.uuid
WHERE %s.name='%s' AND %s
LIMIT ?`, s.embeddingTableName, s.embeddingTableName, s.embeddingTableName,
		s.collectionTableName, s.embeddingTableName, s.collectionTableName, s.collectionTableName, databaseName,
		whereQuery)
	rows, err := s.db.QueryContext(ctx, sql, numDocuments)
	if err != nil {
		return nil, err
	}
	docs := make([]schema.Document, 0)
	defer rows.Close()

	for rows.Next() {
		doc := schema.Document{}
		if err := rows.Scan(&doc.PageContent, &doc.Metadata); err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}
	return docs, rows.Err()
}

func (s Store) DropTables(ctx context.Context) error {
	if _, err := s.db.ExecContext(ctx, fmt.Sprintf(`DROP TABLE IF EXISTS %s`, s.embeddingTableName)); err != nil {
		return err
	}
	if _, err := s.db.ExecContext(ctx, fmt.Sprintf(`DROP TABLE IF EXISTS %s`, s.collectionTableName)); err != nil {
		return err
	}
	return nil
}

func (s Store) RemoveDatabase(ctx context.Context, tx *sql.Tx) error {
	_, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s WHERE name = ?`, s.collectionTableName), s.databaseName)
	return err
}

func (s *Store) createOrGetDatabase(ctx context.Context, tx *sql.Tx) error {
	jsonMetadata, err := json.Marshal(s.databaseMetadata)
	if err != nil {
		return err
	}
	sql := fmt.Sprintf(`INSERT INTO %s (`+"`uuid`"+`, name, cmetadata)
		VALUES(?, ?, ?) ON DUPLICATE KEY UPDATE cmetadata = ?`, s.collectionTableName)
	if _, err := tx.ExecContext(ctx, sql, uuid.New().String(), s.databaseName, jsonMetadata, jsonMetadata); err != nil {
		return err
	}
	sql = fmt.Sprintf(`SELECT `+"`uuid`"+` FROM %s WHERE name = ? ORDER BY name limit 1`, s.collectionTableName)
	return tx.QueryRowContext(ctx, sql, s.databaseName).Scan(&s.databaseUUID)
}

// getOptions applies given options to default Options and returns it
// This uses options pattern so clients can easily pass options without changing function signature.
func (s Store) getOptions(options ...vectorstores.Option) vectorstores.Options {
	opts := vectorstores.Options{}
	for _, opt := range options {
		opt(&opts)
	}
	return opts
}

func (s Store) getDatabaseName(opts vectorstores.Options) string {
	if opts.NameSpace != "" {
		return opts.NameSpace
	}
	return s.databaseName
}

func (s Store) getScoreThreshold(opts vectorstores.Options) (float32, error) {
	if opts.ScoreThreshold < 0 || opts.ScoreThreshold > 1 {
		return 0, ErrInvalidScoreThreshold
	}
	return opts.ScoreThreshold, nil
}

// getFilters return metadata filters, now only support map[key]value pattern
// TODO: should support more types like {"key1": {"key2":"values2"}} or {"key": ["value1", "values2"]}.
func (s Store) getFilters(opts vectorstores.Options) (map[string]any, error) {
	if opts.Filters != nil {
		if filters, ok := opts.Filters.(map[string]any); ok {
			return filters, nil
		}
		return nil, ErrInvalidFilters
	}
	return map[string]any{}, nil
}

func (s Store) deduplicate(
	ctx context.Context,
	opts vectorstores.Options,
	docs []schema.Document,
) []schema.Document {
	if opts.Deduplicater == nil {
		return docs
	}

	filtered := make([]schema.Document, 0, len(docs))
	for _, doc := range docs {
		if !opts.Deduplicater(ctx, doc) {
			filtered = append(filtered, doc)
		}
	}

	return filtered
}
