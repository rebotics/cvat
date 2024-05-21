import { useHistory } from 'react-router';
import React, { useState } from 'react';
import { Col, Row } from 'antd/lib/grid';
import Button from 'antd/lib/button';
import Text from 'antd/lib/typography/Text';
import Input from 'antd/lib/input';

export default function ImageSearchComponent(props: { task: any }): JSX.Element {
    const {
        task: taskInstance,
    } = props;

    const history = useHistory();

    const [imageSearchQuery, setImageSearchQuery] = useState<string>('');

    const foundImages = [];
    if (imageSearchQuery) {
        let frame = 0;
        for (const job of taskInstance.jobs.sort((j) => -j.id)) {
            for (const filename of job.filenames) {
                if (filename.includes(imageSearchQuery)) {
                    foundImages.push({
                        name: filename,
                        jobId: job.id,
                        frame,
                    });
                }
                frame++;
            }
        }
    }

    let searchResults;
    if (imageSearchQuery) {
        if (foundImages.length > 0) {
            searchResults = foundImages.map((item, i) => (
                <Col className='cvat-task-page-image-search-result' xs={24} key={`image-search-result-${i}`}>
                    <Button
                        onClick={
                            () => history.push(`/tasks/${taskInstance.id}/jobs/${item.jobId}/?frame=${item.frame}`)
                        }
                        type='link'
                        size='small'
                    >
                        {item.name}
                    </Button>
                </Col>
            ));
        } else {
            searchResults = (
                <Col className='cvat-task-page-image-search-result' xs={24}>
                    <Text>No images found...</Text>
                </Col>
            );
        }
    } else {
        searchResults = null;
    }

    return (
        <Row>
            <Col xs={24}>
                <Input.Search
                    enterButton
                    onSearch={(phrase: string) => {
                        setImageSearchQuery(phrase);
                    }}
                    defaultValue={imageSearchQuery}
                    className='cvat-task-page-image-search-bar'
                    placeholder='Search image ...'
                />
            </Col>
            {searchResults}
        </Row>
    );
}
